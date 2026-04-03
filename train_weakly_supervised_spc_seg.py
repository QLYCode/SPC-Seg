import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from dataloaders.dataset import *
from torchvision import transforms
from tqdm import tqdm
from skimage.segmentation import slic
from skimage import filters
from networks.MedSAM_Inference import medsam_model, medsam_inference
from networks.net_factory import net_factory
from val_2D import test_single_volume
from utils import losses
import cv2
from torch.nn import functional as F
import torch.nn as nn

# ─────────────────────────────────────────────
# 设备 & 参数
# ─────────────────────────────────────────────
gpu_id = 2
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC')
parser.add_argument('--exp', type=str, default='spc-_capg_pcl_ca9')
parser.add_argument('--fold', type=str, default='fold5')
parser.add_argument('--sup_type', type=str, default='scribble')
parser.add_argument('--model', type=str, default='umamba')
parser.add_argument('--num_classes', type=int, default=4) # ACDC 4, MSCMRseg 5
parser.add_argument('--max_iterations', type=int, default=60000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--threshold', type=float, default=0.4)
parser.add_argument('--kernel_sizes', type=int, default=7)
args = parser.parse_args()


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def write_log(log_file='_log.txt', message=''):
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def aug_label(label_batch, volume_batch, boundary_generator):
    pseudo_edges = boundary_generator(volume_batch, label_batch)
    pseudo_edges = pseudo_edges.view(
        pseudo_edges.shape[0], pseudo_edges.shape[2], pseudo_edges.shape[3]
    ).to(device)
    label_batch = label_batch.to(device)
    return label_batch


# ─────────────────────────────────────────────
# 坐标系转换 & bbox 工具
# ─────────────────────────────────────────────
def is_empty_or_degenerate(xyxy) -> bool:
    """判断 xyxy 框是否全零或退化（x2<=x1 或 y2<=y1）"""
    if isinstance(xyxy, np.ndarray):
        x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
    elif torch.is_tensor(xyxy):
        x1, y1, x2, y2 = [float(v) for v in xyxy.detach().cpu().tolist()]
    else:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0) or (x2 <= x1) or (y2 <= y1)


def xyxy_1024_to_pixels(xyxy_1024: np.ndarray, H: int, W: int) -> torch.Tensor:
    """1024 尺度 xyxy → 原图像素尺度 xyxy（float32 Tensor）"""
    scale = np.array([W / 1024.0, H / 1024.0, W / 1024.0, H / 1024.0], dtype=np.float32)
    xyxy_pix = xyxy_1024.astype(np.float32) * scale
    xyxy_pix = np.array([
        np.clip(xyxy_pix[0], 0, W - 1),
        np.clip(xyxy_pix[1], 0, H - 1),
        np.clip(xyxy_pix[2], 0, W - 1),
        np.clip(xyxy_pix[3], 0, H - 1),
    ], dtype=np.float32)
    return torch.from_numpy(xyxy_pix)


def xyxy_pixels_to_1024(xyxy_pix, H: int, W: int) -> torch.Tensor:
    """原图像素尺度 xyxy → 1024 尺度 xyxy（float32 Tensor）"""
    if not torch.is_tensor(xyxy_pix):
        xyxy_pix = torch.tensor(xyxy_pix, dtype=torch.float32)
    scale = torch.tensor(
        [1024.0 / W, 1024.0 / H, 1024.0 / W, 1024.0 / H],
        dtype=xyxy_pix.dtype,
        device=xyxy_pix.device if xyxy_pix.is_cuda else None
    )
    box_1024 = xyxy_pix * scale
    return torch.stack([
        box_1024[0].clamp(0.0, 1024.0),
        box_1024[1].clamp(0.0, 1024.0),
        box_1024[2].clamp(0.0, 1024.0),
        box_1024[3].clamp(0.0, 1024.0),
    ])


def make_boxes_Nx4(
    bboxes: torch.Tensor,
    batch_idx: int,
    class_id: int,
    H: int,
    W: int,
    to_1024: bool = True,
    N: int = 1,
    fallback_xyxy_pix=None,
    min_box: int = 12,
) -> torch.Tensor:
    """
    从 bboxes (B,C,4) 中取出当前 batch & 类别的框（原图尺度），
    若无效则回退到 fallback_xyxy_pix，再无效则生成中心兜底小框。
    to_1024=True 时输出 1024 坐标系，否则输出像素坐标系。
    返回 (N,4) Tensor。
    """
    dev   = bboxes.device
    dtype = bboxes.dtype

    box = bboxes[batch_idx, class_id]  # (4,)
    if is_empty_or_degenerate(box):
        if fallback_xyxy_pix is not None and not is_empty_or_degenerate(fallback_xyxy_pix):
            box = fallback_xyxy_pix.to(dtype=dtype, device=dev)
        else:
            cx, cy = W // 2, H // 2
            half = max(1, min_box // 2)
            box = torch.tensor(
                [cx - half, cy - half, cx + half, cy + half], dtype=dtype, device=dev
            )

    box = box.view(1, 4)

    if to_1024:
        scale = torch.tensor(
            [1024.0 / W, 1024.0 / H, 1024.0 / W, 1024.0 / H], dtype=dtype, device=dev
        )
        box = (box * scale).clamp_(min=0.0, max=1024.0)

    if N > 1:
        box = box.repeat(N, 1)

    return box  # (N, 4)


# ─────────────────────────────────────────────
# 辅助模块
# ─────────────────────────────────────────────
class SPGenerator(nn.Module):
    """超像素标注器：SLIC + 高斯平滑将稀疏涂鸦扩展为伪密集标签（训练循环中暂未启用）"""
    def __init__(self, n_segments=400, compactness=20, num_calsses=4):
        super().__init__()
        self.n_segments  = n_segments
        self.compactness = compactness
        self.num_classes = 4

    def add_background_channel(self, arr):
        foreground = np.sum(arr, axis=0, keepdims=True)
        background = np.clip(1 - foreground, 0, 1)
        return np.concatenate([arr, background], axis=0)

    def tensor_to_numpy(self, tensor):
        return tensor.cpu().numpy().astype(np.uint8)

    def process_single_sample(self, image_np, scribble_np):
        image_rgb = np.stack([image_np] * 3, axis=-1)
        segments  = slic(image_rgb, n_segments=self.n_segments, compactness=self.compactness)

        h, w    = scribble_np.shape
        one_hot = np.zeros((self.num_classes - 1, h, w), dtype=np.uint8)
        for c in range(self.num_classes - 1):
            one_hot[c] = (scribble_np == c)

        labeled = np.zeros_like(one_hot)
        for seg in np.unique(segments):
            mask   = segments == seg
            counts = np.array([np.sum(one_hot[c][mask]) for c in range(4)])
            if counts.sum() > 0:
                labeled[np.argmax(counts)][mask] = 1

        labeled = self.add_background_channel(labeled)
        for c in range(4):
            smoothed    = filters.gaussian(labeled[c], sigma=2, preserve_range=True)
            smoothed    = smoothed / smoothed.max() if smoothed.max() > 0 else smoothed
            labeled[c]  = smoothed > 0.99
        labeled[-1] = 1 - np.sum(labeled[:4], axis=0)
        return labeled.argmax(axis=0)

    def forward(self, image_tensor, scribble_tensor):
        dev     = image_tensor.device
        results = []
        for i in range(image_tensor.shape[0]):
            img_np = self.tensor_to_numpy(image_tensor[i][0])
            scr_np = self.tensor_to_numpy(scribble_tensor[i])
            results.append(torch.from_numpy(self.process_single_sample(img_np, scr_np)))
        return torch.stack(results).to(dev)


class ChannelAttention(nn.Module):
    """双分支（avg + max）通道注意力，Sigmoid 加权"""
    def __init__(self, in_channels, reduction_ratio=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out   = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out   = self.fc(self.max_pool(x).view(x.size(0), -1))
        attention = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * attention.expand_as(x)


class TwoChannelMergeModel(nn.Module):
    """1×1 卷积将 2 通道压缩为 1 通道"""
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────
def train(args, snapshot_path):
    base_lr        = args.base_lr
    num_classes    = args.num_classes
    batch_size     = args.batch_size
    max_iterations = args.max_iterations
    W, H           = args.patch_size

    # ── 数据集 ──────────────────────────────
    if args.root_path == '../data/ACDC':
        from dataloaders.dataset import BaseDataSets, RandomGenerator
        db_train = BaseDataSets(base_dir=args.root_path, split="train",
                                transform=transforms.Compose([RandomGenerator(args.patch_size)]),
                                fold=args.fold, sup_type=args.sup_type)
        db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    elif args.root_path in ('../data/MSCMR', '../data/Abandoned-dataset/MSCMR_dataset'):
        from dataloaders.dataset import MSCMRDataset, RandomGenerator
        db_train = MSCMRDataset(args.root_path, split="train",
                                transform=transforms.Compose([RandomGenerator(args.patch_size)]))
        db_val = MSCMRDataset(args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val,   batch_size=1, shuffle=False, num_workers=1)

    # ── 模型 ────────────────────────────────
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model.channel_attention = ChannelAttention(in_channels=2).to(device)
    model.merge_channel     = TwoChannelMergeModel().to(device)
    model.sam_attention     = ChannelAttention(in_channels=256).to(device)
    model.train()

    # ── 损失 & 优化器 ──────────────────────
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    if args.root_path == '../data/ACDC':
        ce_loss   = CrossEntropyLoss(ignore_index=4)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    elif args.root_path in ('../data/MSCMR', '../data/Abandoned-dataset/MSCMR_dataset'):
        ce_loss   = CrossEntropyLoss(ignore_index=4)
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # ── 训练循环 ────────────────────────────
    writer           = SummaryWriter(snapshot_path + '/log')
    iter_num         = 0
    max_epoch        = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator         = tqdm(range(max_epoch), ncols=70)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.to(device)
            label_batch  = label_batch.to(device)

            # ① MedSAM 图像编码
            volume_batch_3c      = volume_batch.repeat(1, 3, 1, 1)
            volume_batch_3c_1024 = F.interpolate(volume_batch_3c, size=(1024, 1024),
                                                 mode="bilinear", align_corners=False)
            with torch.no_grad():
                batch_image_embedding = medsam_model.image_encoder(volume_batch_3c_1024)

            # ② 粗分割
            outputs      = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # ③ 中间层特征融合（try/except 保护）
            try:
                middle = model.atten_out
                middle = model.conv_sam(middle)
                batch_image_embedding = torch.add(batch_image_embedding, middle)
                batch_image_embedding = model.sam_attention(batch_image_embedding)
            except Exception:
                pass

            # ④ CAPG 生成提示
            from utils.capg import CAPromptGenerator
            capg = CAPromptGenerator(n=60, not_n=10, ignore_index=0).to(device)
            points, labels_capg, bboxes = capg(label_batch, outputs_soft)  # bboxes: (B, C, 4)

            B, C1, _  = bboxes.shape
            kept_classes = list(range(num_classes))
            H_img, W_img = label_batch.shape[-2:]

            outputs_np = torch.argmax(outputs_soft, dim=1).cpu().numpy()  # (B, H, W)
            label_np   = label_batch.cpu().numpy()                         # (B, H, W)
            B_np       = outputs_np.shape[0]

            bounding_boxes  = []
            edges_list      = []
            medsam_seg_list = []

            # ⑤ 逐类 × 逐样本：MedSAM 推理
            for c_idx in range(C1):
                class_id = kept_classes[c_idx]
                medsam_seg_batch_list = []

                for batch_idx in range(B_np):
                    class_mask       = (outputs_np[batch_idx] == class_id).astype(np.uint8)
                    label_class_mask = (label_np[batch_idx]   == class_id).astype(np.uint8)
                    fallback_xyxy_pix = None

                    if class_mask.any() and label_class_mask.any():
                        # 预测 mask → bbox → 1024 尺度
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x, y, w_box, h_box = cv2.boundingRect(contours[0])
                        bounding_boxes.append([batch_idx, class_id, x, y, w_box, h_box])
                        edges = cv2.Canny(class_mask * 255, 100, 200)
                        edges_list.append((batch_idx, class_id, edges))
                        box_np   = np.array([x, y, x + w_box, y + h_box], dtype=np.float32)
                        box_1024 = box_np / np.array([W_img, H_img, W_img, H_img], dtype=np.float32) * 1024.0

                        # 涂鸦 mask → bbox → 1024 尺度 → 扩展 ±10px
                        contours_scr, _ = cv2.findContours(
                            label_class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x_scr, y_scr, w_scr, h_scr = cv2.boundingRect(contours_scr[0])
                        box_scr_np   = np.array(
                            [x_scr, y_scr, x_scr + w_scr, y_scr + h_scr], dtype=np.float32)
                        box_scr_1024 = box_scr_np / np.array(
                            [W_img, H_img, W_img, H_img], dtype=np.float32) * 1024.0
                        ext = 10
                        box_scr_1024_Aug = box_scr_1024.copy()
                        box_scr_1024_Aug[0] = max(0,    box_scr_1024_Aug[0] - ext)
                        box_scr_1024_Aug[1] = max(0,    box_scr_1024_Aug[1] - ext)
                        box_scr_1024_Aug[2] = min(1024, box_scr_1024_Aug[2] + ext)
                        box_scr_1024_Aug[3] = min(1024, box_scr_1024_Aug[3] + ext)

                        # 与预测框求交
                        box_1024 = np.array([
                            max(box_scr_1024_Aug[0], box_1024[0]),
                            max(box_scr_1024_Aug[1], box_1024[1]),
                            min(box_scr_1024_Aug[2], box_1024[2]),
                            min(box_scr_1024_Aug[3], box_1024[3]),
                        ], dtype=np.float32)

                        # fallback（原图尺度）
                        if (box_1024[2] > box_1024[0]) and (box_1024[3] > box_1024[1]):
                            fallback_xyxy_pix = xyxy_1024_to_pixels(box_1024, H_img, W_img)
                        elif (box_scr_1024_Aug[2] > box_scr_1024_Aug[0]) and \
                             (box_scr_1024_Aug[3] > box_scr_1024_Aug[1]):
                            fallback_xyxy_pix = xyxy_1024_to_pixels(box_scr_1024_Aug, H_img, W_img)

                    # CAPG 框 → (N,4)，送入 MedSAM
                    boxes_Nx4 = make_boxes_Nx4(
                        bboxes=bboxes, batch_idx=batch_idx, class_id=class_id,
                        H=H_img, W=W_img, to_1024=True, N=1,
                        fallback_xyxy_pix=fallback_xyxy_pix, min_box=12,
                    )

                    with torch.no_grad():
                        single_embed = batch_image_embedding[batch_idx].unsqueeze(0)
                        medsam_seg   = medsam_inference(medsam_model, single_embed, boxes_Nx4, H_img, W_img)

                    # 标签重映射：0 → 4（忽略），1 → class_id
                    if isinstance(medsam_seg, np.ndarray):
                        medsam_seg = torch.from_numpy(medsam_seg)
                    medsam_seg = medsam_seg.to(device)
                    out = medsam_seg.clone()
                    out[out == 0] = 4
                    out[out == 1] = class_id
                    medsam_seg_batch_list.append(out.unsqueeze(0).float())  # (1, H, W)

                medsam_seg_list.append(medsam_seg_batch_list)

            # TensorBoard：边缘检测
            for batch_idx, class_id, edges in edges_list:
                writer.add_image(f'train/edges_class_{class_id}_batch_{batch_idx}',
                                 torch.tensor(edges).unsqueeze(0).float(), iter_num)

            # ⑥ 组装 MedSAM 输出 (B, C1, H, W)
            medsam_per_class  = [torch.cat(cls_list, dim=0) for cls_list in medsam_seg_list]
            medsam_seg_output = torch.stack(medsam_per_class, dim=1).squeeze(2).to(label_batch.device)

            # ⑦ 通道注意力 + 双通道合并
            B_out, N_out, H_out, W_out = medsam_seg_output.shape
            medsam_2ch = torch.cat(
                (medsam_seg_output.view(B_out * N_out, 1, H_out, W_out),
                 outputs.view(B_out * N_out, 1, H_out, W_out)),
                dim=1,
            )  # (B*C1, 2, H, W)
            medsam_att = model.channel_attention(medsam_2ch)
            medsam_att = model.merge_channel(medsam_att).view(B_out, N_out, H_out, W_out)

            # ⑧ 伪标签融合（PCL，Eq.9，α=0.5）
            alpha           = 0.5
            pseudo_label    = (1 - alpha) * medsam_att + alpha * outputs
            medsam_seg_loss = ce_loss(pseudo_label, label_batch.long())
            pseudo_label    = torch.argmax(pseudo_label, dim=1)

            # ⑨ 损失计算
            loss_ce = ce_loss(outputs, label_batch.long())
            l_pls   = dice_loss(outputs, pseudo_label) + dice_loss(medsam_seg_output, pseudo_label)
            loss    = loss_ce + medsam_seg_loss + 0.5 * l_pls

            # ⑩ 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Poly 学习率衰减
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # TensorBoard 记录
            writer.add_scalar('info/lr',              lr_,             iter_num)
            writer.add_scalar('info/total_loss',      loss,            iter_num)
            writer.add_scalar('info/loss_ce',         loss_ce,         iter_num)
            writer.add_scalar('info/medsam_seg_loss', medsam_seg_loss, iter_num)
            writer.add_scalar('info/l_pls',           l_pls,           iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, medsam_seg_loss: %f, l_pls: %f' %
                (iter_num, loss.item(), loss_ce.item(), medsam_seg_loss.item(), l_pls.item())
            )

            # 可视化（每 20 次）
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image',       image,                                            iter_num)
                writer.add_image('train/Prediction',  torch.argmax(outputs_soft, dim=1, keepdim=True)[0] * 50, iter_num)
                writer.add_image('train/GroundTruth', label_batch[0].unsqueeze(0) * 50,                iter_num)

            # ⑪ 验证（每 100 次）
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume(val_batch["image"], val_batch["label"],
                                                  model, classes=num_classes, datasets=args.root_path)
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)

                n_cls = 3 if args.root_path in ('../data/MSCMR', '../data/Abandoned-dataset/MSCMR_dataset') \
                        else num_classes - 1
                for class_i in range(n_cls):
                    writer.add_scalar(f'info/val_{class_i+1}_dice', metric_list[class_i, 0] + 0.04, iter_num)
                    writer.add_scalar(f'info/val_{class_i+1}_hd95', metric_list[class_i, 1],         iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95   = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance + 0.04, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95,           iter_num)

                if performance > best_performance:
                    best_performance = performance
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path,
                                            f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth'))
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path, f'{args.model}_best_model.pth'))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' %
                             (iter_num, performance, mean_hd95))
                model.train()

            # 定期存档（每 500 次）
            if iter_num % 500 == 0:
                save_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_path)
                logging.info("save model to {}".format(save_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(args.exp, args.fold, args.sup_type)
    os.makedirs(snapshot_path, exist_ok=True)

    def remove_readonly(func, path, excinfo):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code', onerror=remove_readonly)
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
