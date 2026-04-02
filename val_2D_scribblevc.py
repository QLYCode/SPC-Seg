import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import pandas as pd
from torchmetrics.classification import MultilabelAccuracy

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
catagory_list = pd.read_excel('../data/ACDC/slice_classification.xlsx')
catagory_list.set_index('slice', inplace=True)
catagory_list = catagory_list.astype(bool)

test_accuracy = MultilabelAccuracy(num_labels=4).cuda()


# ─────────────────────────────────────────────
# 单样本指标计算
# ─────────────────────────────────────────────
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan, np.nan
    elif gt.sum() == 0 and pred.sum() > 0:
        return 0, 0
    elif gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = np.nan if pred.sum() == 0 else metric.binary.hd95(pred, gt)
        return dice, hd95


# ─────────────────────────────────────────────
# 标准单卷评估（单输出网络）
# ─────────────────────────────────────────────
def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y   = slice_.shape
            slice_  = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out  = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
                pred = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out        = torch.argmax(torch.softmax(net(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


# ─────────────────────────────────────────────
# CAM 双辅助输出网络评估
# ─────────────────────────────────────────────
@torch.no_grad()
def test_single_volume_CAM(image, label, net, classes, patch_size=[256, 256], epoch=None, model_type=None):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y   = slice_.shape
            slice_  = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_aux1, out_aux2, cls_output = net(inp, ep=epoch, model_type=model_type)
                out_aux1_soft = torch.softmax(out_aux1, dim=1)
                out_aux2_soft = torch.softmax(out_aux2, dim=1)
                out  = torch.argmax(
                    (torch.min(out_aux1_soft, out_aux2_soft) > 0.5) *
                    (0.5 * out_aux1_soft + 0.5 * out_aux2_soft),
                    dim=1
                ).squeeze(0)
                pred = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).float().cuda()
        net.eval()
        with torch.no_grad():
            out_aux1_soft = torch.softmax(net(inp)[0], dim=1)
            out_aux2_soft = torch.softmax(net(inp)[1], dim=1)
            out        = torch.argmax(
                (torch.min(out_aux1_soft, out_aux2_soft) > 0.5) *
                (0.5 * out_aux1_soft + 0.5 * out_aux2_soft),
                dim=1
            ).squeeze(0)
            prediction = zoom(out.cpu().detach().numpy(), (x / patch_size[0], y / patch_size[1]), order=0)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list