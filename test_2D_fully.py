import argparse
import os
import re
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.dataset import *
from networks.net_factory import net_factory
from train_weakly_supervised_BASELINE_PCL import TwoChannelMergeModel
from train_weakly_supervised_SparseMamba_PCL import ChannelAttention

# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Abandoned-dataset/MSCMR_dataset',
                    help='Name of Experiment')
# ACDC      '../data/ACDC'
# chaos     '../data/CHAOs'
# MSCMR    '../data/Abandoned-dataset/MSCMR_dataset'
parser.add_argument('--exp', type=str,
                    default='MSCMR_pcl_fold5', help='experiment_name')
# unet ccnet alignseg bisenet mobilenet ecanet efficientumamba segmamba
# umamba swinumamba lkmunet segmamba segmenter unetformer segformer sparsemamba
parser.add_argument('--model', type=str,
                    default='umamba', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold5', help='fold')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='label')

args = parser.parse_args()

# ─────────────────────────────────────────────
# 设备
# ─────────────────────────────────────────────
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 折叠划分
# ─────────────────────────────────────────────
def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]

    fold1_testing_set  = ["patient{:0>3}".format(i) for i in range(1,  21)]
    fold2_testing_set  = ["patient{:0>3}".format(i) for i in range(21, 41)]
    fold3_testing_set  = ["patient{:0>3}".format(i) for i in range(41, 61)]
    fold4_testing_set  = ["patient{:0>3}".format(i) for i in range(61, 81)]
    fold5_testing_set  = ["patient{:0>3}".format(i) for i in range(81, 101)]

    fold_map = {
        "fold1": fold1_testing_set,
        "fold2": fold2_testing_set,
        "fold3": fold3_testing_set,
        "fold4": fold4_testing_set,
        "fold5": fold5_testing_set,
    }

    if fold not in fold_map:
        return "ERROR KEY"

    testing_set  = fold_map[fold]
    training_set = [i for i in all_cases_set if i not in testing_set]
    return [training_set, testing_set]


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
        # gt 全零：hd95 取 256×256 下的最大边界距离
        return 0, float(181.02)
    else:
        return 0, 0


# ─────────────────────────────────────────────
# 单卷评估
# ─────────────────────────────────────────────
def test_single_volume(img, lab, net, classes, patch_size=[256, 256], datasets=None):
    image = img.cpu().squeeze(0).numpy()
    label = lab.cpu().squeeze(0).numpy()

    if len(image.shape) == 3:
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
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list


# ─────────────────────────────────────────────
# 推理主流程
# ─────────────────────────────────────────────
def Inference(FLAGS):
    if args.root_path == '../data/ACDC':
        train_ids, test_ids = get_fold_ids(FLAGS.fold)
        all_volumes = os.listdir(FLAGS.root_path + "/ACDC_training_volumes")
    elif args.root_path in ('../data/MSCMR', '../data/Abandoned-dataset/MSCMR_dataset'):
        all_volumes      = os.listdir(FLAGS.root_path + "/TestSet/images")
        all_volumes_path = FLAGS.root_path + "/TestSet/images"
        all_batches_path = FLAGS.root_path + "/TestSet/labels"
        test_ids         = all_volumes

    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, all_volumes))
        image_list.extend(new_data_list)

    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.sup_type)
    test_save_path = "../model/{}/{}/{}_predictions/".format(FLAGS.exp, FLAGS.sup_type, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    # 模型构建 & 权重加载
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    net.channel_attention = ChannelAttention(in_channels=2).to(device)
    net.merge_channel     = TwoChannelMergeModel().to(device)
    net.sam_attention     = ChannelAttention(in_channels=256).to(device)

    save_mode_path = os.path.join(snapshot_path, 'umamba_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    print(len(image_list))

    # 测试集推理
    db_test    = MSCMRDataset(args.root_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume(
            sampled_batch["image"], sampled_batch["label"],
            net, classes=5, datasets=args.root_path
        )
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95   = np.mean(metric_list, axis=0)[1]

    return performance, mean_hd95


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    mean_dice, mean_hd95 = Inference(FLAGS)
    print(mean_dice, mean_hd95)