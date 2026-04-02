import random

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def show_image(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).show()

def save_image(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)

def show_rgb(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).show()

def to_onehot(tensor, num_classes=None, ignore_index=None):
    """
    tensor: [B, C, H, W] 的 logits/概率（或同形状的张量）
    num_classes: 类别数；默认自动取 C
    ignore_index: 可选（一般用于标签，这里只是兜底）
    return: [B, num_classes, H, W] float
    """
    # 兼容 numpy
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if tensor.dim() != 4:
        raise ValueError(f"to_onehot expects 4D [B,C,H,W], got {tuple(tensor.shape)}")

    if num_classes is None:
        num_classes = tensor.shape[1]  # 自动用通道数

    winning = torch.argmax(tensor, dim=1).long()  # [B,H,W]

    C = tensor.shape[1]
    if num_classes < C:
        raise ValueError(f"num_classes({num_classes}) < tensor.channels({C}); "
                         f"set num_classes={C} or leave it as None")

    one_hot = F.one_hot(winning, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if ignore_index is not None:
        mask = (winning == ignore_index).unsqueeze(1)  # [B,1,H,W]
        if mask.any():
            one_hot = one_hot.masked_fill(mask.expand_as(one_hot), 0.0)

    # 不做 .cpu().numpy()，保持 torch；如需可视化/导出再转换
    return one_hot