import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

# ─────────────────────────────────────────────
# 设备
# ─────────────────────────────────────────────
gpu_id = 2
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 单样本指标计算
# ─────────────────────────────────────────────
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif gt.sum() == 0:
        # gt 全零：Dice=0，hd95 取 256×256 下的最大边界距离
        return 0, float(181.02)
    else:
        return 0, 0


# ─────────────────────────────────────────────
# 标准单卷评估（单输出网络）
# ─────────────────────────────────────────────
def test_single_volume(image, label, net, classes, patch_size=[256, 256], datasets=None):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        # 3D 卷：逐切片推理
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y   = slice_.shape
            slice_  = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                out  = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
                pred = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        # 2D 单张推理
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out        = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # 按数据集选取评估类别范围
    metric_list = []
    if datasets == '../data/CHAOs':
        # CHAOS：类别 0 也是目标分类
        for i in range(0, classes - 1):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
    elif datasets in ('../data/MSCMR', '../data/Abandoned-dataset/MSCMR_dataset'):
        for i in range(1, classes - 1):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
    else:
        # ACDC 及其他
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list


# ─────────────────────────────────────────────
# 深监督网络评估（四输出，取第一个）
# ─────────────────────────────────────────────
def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y   = slice_.shape
            slice_  = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(inp)
                out  = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
                pred = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(inp)
            out        = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


# ─────────────────────────────────────────────
# CCT 网络评估（双输出，取第一个）
# ─────────────────────────────────────────────
def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y   = slice_.shape
            slice_  = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                output_main = net(inp)[0]
                out  = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
                pred = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            output_main, _ = net(inp)
            out        = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list