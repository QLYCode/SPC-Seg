import argparse
from random import random
import numpy as np
from torch.backends import cudnn
import torch.nn as nn
import torch

# ─────────────────────────────────────────────
# 设备
# ─────────────────────────────────────────────
gpu_id = 2
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 网络工厂
# ─────────────────────────────────────────────
def net_factory(net_type="unet", in_chns=1, class_num=3):

    # ── UNet 系列 ──────────────────────────────────────────
    if net_type == "unet":
        from networks.unet import UNet
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)

    elif net_type == "unet_cct":
        from networks.unet_cct import UNet_CCT
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).to(device)

    elif net_type == "unet_cct_3h":
        from networks.unet import UNet_CCT_3H
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).to(device)

    elif net_type == "unet_ds":
        from networks.unet import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num).to(device)

    elif net_type == "efficient_unet":
        from networks.efficientunet import Effi_UNet
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).to(device)

    elif net_type == "pnet":
        from networks.pnet import PNet2D
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).to(device)

    # ── CNN 系列 ───────────────────────────────────────────
    elif net_type == "ghostnet":
        from networks.ghostnet import ghostnet
        net = ghostnet().to(device)

    elif net_type == "ccnet":
        from networks.ccnet import CCNet_Model
        net = CCNet_Model(num_classes=class_num).to(device)

    elif net_type == "alignseg":
        from networks.alignseg import AlignSeg
        net = AlignSeg(num_classes=class_num).to(device)

    elif net_type == "bisenet":
        from networks.bisenet import BiSeNet
        net = BiSeNet(in_channel=in_chns, nclass=class_num, backbone='resnet18').to(device)

    elif net_type == "mobilenet":
        from networks.mobilenet import MobileNetV2
        net = MobileNetV2(input_channel=in_chns, n_class=class_num).to(device)

    elif net_type == "ecanet":
        from networks.ecanet import ECA_MobileNetV2
        net = ECA_MobileNetV2(n_class=4, width_mult=1).to(device)

    # ── Transformer 系列 ───────────────────────────────────
    elif net_type == "segmenter":
        from networks.segmenter import create_segmenter
        model_cfg = {
            "image_size": (256, 256),
            "patch_size": 16,
            "d_model": 192,
            "n_heads": 3,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
            "backbone": "vit_tiny_patch16_384",
            "decoder": {
                "name": "mask_transformer",
                "drop_path_rate": 0.0,
                "dropout": 0.1,
                "n_layers": 2,
            },
            "n_cls": class_num,
        }
        net = create_segmenter(model_cfg).to(device)

    elif net_type == "unetformer":
        from networks.unetformer import UNetFormer
        net = UNetFormer(num_classes=class_num).to(device)

    elif net_type == "segformer":
        from networks.segformer import SegFormer
        net = SegFormer(num_classes=class_num).to(device)

    elif net_type == "swin_transformer":
        from networks.swin_transformer import SwinTransformer
        net = SwinTransformer().to(device)

    elif net_type == "swinunet":
        from networks.swin_unet import get_swin_unet
        net = get_swin_unet().to(device)

    elif net_type == "transunet":
        from networks.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

        parser = argparse.ArgumentParser()
        parser.add_argument('--root_path', type=str,
                            default='../data/Synapse/train_npz')
        parser.add_argument('--dataset', type=str, default='ACDC')
        parser.add_argument('--list_dir', type=str,
                            default='./lists/lists_Synapse')
        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--max_iterations', type=int, default=30000)
        parser.add_argument('--max_epochs', type=int, default=150)
        parser.add_argument('--batch_size', type=int, default=24)
        parser.add_argument('--n_gpu', type=int, default=1)
        parser.add_argument('--deterministic', type=int, default=1)
        parser.add_argument('--base_lr', type=float, default=0.01)
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--n_skip', type=int, default=3)
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
        parser.add_argument('--vit_patches_size', type=int, default=16)
        args = parser.parse_args()

        if not args.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        args.num_classes  = 4
        args.root_path    = "../data/ACDC"
        args.list_dir     = "../data/train.list"
        args.is_pretrain  = True
        args.exp          = 'transunet' + args.dataset + str(args.img_size)

        config_vit           = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip    = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size),
                int(args.img_size / args.vit_patches_size)
            )
        net = ViT_seg(config_vit, img_size=args.img_size,
                      num_classes=config_vit.n_classes).to(device)

    # ── Mamba 系列 ─────────────────────────────────────────
    elif net_type == "msvmunet":
        from networks.msvmunet import build_msvmunet_model
        net = build_msvmunet_model(in_channels=3, num_classes=class_num).to(device)

    elif net_type == "segmamba":
        from networks.segmamba import SegMamba
        net = SegMamba(in_chns, class_num).to(device)

    elif net_type == "swinumamba":
        from networks.swinumamba import SwinUMamba
        net = SwinUMamba(in_chns, class_num).to(device)

    elif net_type == "emnet":
        from networks.em_net_model import EMNet
        net = EMNet(in_chns, class_num).to(device)

    elif net_type == "umamba":
        from networks.umambabot_2d import UMambaBot
        net = UMambaBot(
            input_channels=in_chns,
            n_stages=4,                              # 包含 stem 的 4 个编码阶段
            features_per_stage=[32, 64, 128, 256],  # 各阶段特征通道数
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3]],
            strides=[1, 2, 2, 2],                   # 下采样策略（第一次保持分辨率）
            n_conv_per_stage=[2, 2, 2, 2],           # 编码器各阶段卷积层数
            num_classes=class_num,
            n_conv_per_stage_decoder=[2, 2, 2],      # 解码器各阶段卷积层数（比编码器少1阶）
            conv_bias=False,
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        ).to(device)

    elif net_type == "lmunet":
        from networks.lkmunet import LKMUNet
        net = LKMUNet(in_channels=in_chns, out_channels=class_num,
                      kernel_sizes=[21, 15, 9, 3]).to(device)

    # ── 自研网络 ───────────────────────────────────────────
    elif net_type == "sparsemamba":
        from networks.umamba_e2sd import UMambaBot
        net = UMambaBot(
            input_channels=in_chns,
            n_stages=4,
            features_per_stage=[32, 64, 128, 256],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3]],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[2, 2, 2, 2],
            num_classes=class_num,
            n_conv_per_stage_decoder=[2, 2, 2],
            conv_bias=False,
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        ).to(device)

    else:
        net = None

    return net


# ─────────────────────────────────────────────
# FLOPs / Params 测试入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from thop import profile

    model = net_factory("unet", in_chns=1, class_num=4)
    inp   = torch.randn(1, 1, 256, 256).to(device)
    model = model.to(device)

    for name, param in model.named_parameters():
        print(name, param.dtype)
        break  # 只看第一层

    flops, params = profile(model, inputs=(inp,))
    print(f"FLOPs: {flops}, Params: {params}")