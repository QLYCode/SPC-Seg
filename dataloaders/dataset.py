import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pydicom
import nibabel as nib

from random import sample
from scipy.ndimage import zoom, gaussian_filter, map_coordinates

def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.all_volumes = [i for i in self.all_volumes if '.h5' in i]
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        elif fold == "MAAGfold":
            training_set = ["patient{:0>3}".format(i) for i in
                            [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                             71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in
                            [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                             71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51,
                             40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                             23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = sample(["patient{:0>3}".format(i) for i in
                                   [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                    71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3,
                                    8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                                    23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

# class RandomGenerator_chaos(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         # ind = random.randrange(0, img.shape[0])
#         # image = img[ind, ...]
#         # label = lab[ind, ...]
#         if random.random() < 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             if 4 in np.unique(label):
#                 image, label = random_rotate(image, label, cval=4)
#             else:
#                 image, label = random_rotate(image, label, cval=0)
#         x, y = image.shape
#         image = zoom(
#             image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         label = zoom(
#             label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(
#             image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.uint8))
#         sample = {'image': image, 'label': label}
#         return sample

class RandomGenerator_chaos(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机旋转和翻转
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)

        # 新增随机缩放
        elif random.random() < 0.5:
            h, w = image.shape
            scale_factor = random.uniform(0.85, 1.15)  # 缩放范围0.85-1.15
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # 确保缩放后至少1像素
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            
            # 使用最近邻插值保持标签准确性
            image_scaled = zoom(image, (new_h/h, new_w/w), order=0)
            label_scaled = zoom(label, (new_h/h, new_w/w), order=0)
            
            # 填充或裁剪回原始尺寸
            if scale_factor < 1.0:  # 缩小：填充
                pad_h = h - new_h
                pad_w = w - new_w
                # 均匀填充
                image = np.pad(image_scaled, ((pad_h//2, pad_h - pad_h//2),
                                             (pad_w//2, pad_w - pad_w//2)),
                              mode='constant', constant_values=4)
                label = np.pad(label_scaled, ((pad_h//2, pad_h - pad_h//2),
                                             (pad_w//2, pad_w - pad_w//2)),
                              mode='constant', constant_values=4)
            else:  # 放大：随机裁剪
                start_h = random.randint(0, new_h - h)
                start_w = random.randint(0, new_w - w)
                image = image_scaled[start_h:start_h+h, start_w:start_w+w]
                label = label_scaled[start_h:start_h+h, start_w:start_w+w]

        # 最终缩放到目标尺寸
        x, y = image.shape
        image = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        label = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)

        # 转换为Tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        return {'image': image, 'label': label}
    


# class RandomGenerator_chaos(object):
#     def __init__(self, output_size):
#         self.output_size = output_size
#         self.background_class = 4  # 明确背景类别
        
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
        
#         # 并行执行多种增强（非互斥）
#         image, label = self.geometric_aug(image, label)
#         image, label = self.scale_crop_aug(image, label)
#         image, label = self.copy_paste_aug(image, label)
#         image = self.intensity_aug(image)
        
#         # 最终尺寸调整
#         image, label = self.final_resize(image, label)
        
#         return {
#             'image': torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
#             'label': torch.from_numpy(label.astype(np.uint8))
#         }

#     def geometric_aug(self, img, lbl):
#         """组合几何变换"""
#         # 随机旋转（0-360度连续）
#         if random.random() < 0.7:
#             angle = random.uniform(-15, 15)
#             img, lbl = self.rotate(img, lbl, angle)
            
#         # 随机翻转
#         if random.random() < 0.5:
#             axis = random.randint(0, 1)
#             img = np.flip(img, axis=axis).copy()
#             lbl = np.flip(lbl, axis=axis).copy()
            
#         # 弹性变形
#         if random.random() < 0.3:
#             img, lbl = self.elastic_transform(img, lbl)
            
#         return img, lbl
    
#     def rotate(self, img, lbl, angle):
#         """旋转增强（保持图像中心）"""
#         center = (img.shape[1]//2, img.shape[0]//2)
#         rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#         img = cv2.warpAffine(img, rot_mat, img.shape[::-1], 
#                            flags=cv2.INTER_LINEAR,
#                            borderMode=cv2.BORDER_CONSTANT,
#                            borderValue=self.background_class)
#         lbl = cv2.warpAffine(lbl, rot_mat, lbl.shape[::-1], 
#                             flags=cv2.INTER_NEAREST,
#                             borderMode=cv2.BORDER_CONSTANT,
#                             borderValue=self.background_class)
#         return img, lbl
    
#     def elastic_transform(self, img, lbl, alpha=200, sigma=20):
#         """弹性变形（医学影像优化参数）"""
#         random_state = np.random.RandomState(None)
#         shape = img.shape
        
#         dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

#         x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#         indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        
#         dist_img = map_coordinates(img, indices, order=1, mode='constant', cval=self.background_class).reshape(shape)
#         dist_lbl = map_coordinates(lbl, indices, order=0, mode='constant', cval=self.background_class).reshape(shape)
#         return dist_img, dist_lbl
    
#     def scale_crop_aug(self, img, lbl):
#         """改进的缩放裁剪策略"""
#         if random.random() < 0.5:
#             scale = random.choice([0.8, 0.9, 1.1, 1.2])
#             h, w = img.shape
            
#             # 缩放
#             new_h, new_w = int(h*scale), int(w*scale)
#             img_scaled = zoom(img, (new_h/h, new_w/w), order=3)
#             lbl_scaled = zoom(lbl, (new_h/h, new_w/w), order=0)
            
#             # 裁剪或填充
#             if scale < 1:
#                 pad_h = h - new_h
#                 pad_w = w - new_w
#                 img = np.pad(img_scaled, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
#                             mode='constant', constant_values=self.background_class)
#                 lbl = np.pad(lbl_scaled, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
#                             mode='constant', constant_values=self.background_class)
#             else:
#                 # 智能裁剪：优先保留目标区域
#                 y_indices, x_indices = np.where(lbl < 4)
#                 if len(y_indices) > 0:
#                     y_min, y_max = np.min(y_indices), np.max(y_indices)
#                     x_min, x_max = np.min(x_indices), np.max(x_indices)
#                     y_start = max(0, min(y_min, new_h - h))
#                     x_start = max(0, min(x_min, new_w - w))
#                 else:
#                     y_start = random.randint(0, new_h - h)
#                     x_start = random.randint(0, new_w - w)
#                 img = img_scaled[y_start:y_start+h, x_start:x_start+w]
#                 lbl = lbl_scaled[y_start:y_start+h, x_start:x_start+w]
                
#         return img, lbl
    
#     def copy_paste_aug(self, img, lbl):
#         """自包含的Copy-Paste增强（无需外部对象库）"""
#         if random.random() < 0.3:
#             # 在本图像内复制对象
#             obj_classes = [c for c in range(4) if np.any(lbl == c)]
#             if not obj_classes:
#                 return img, lbl
                
#             selected_class = random.choice(obj_classes)
#             mask = (lbl == selected_class).astype(np.uint8)
            
#             if np.sum(mask) > 10:
#                 # 寻找粘贴位置（背景区域）
#                 bg_mask = (lbl == self.background_class)
#                 if np.sum(bg_mask) < 10:
#                     return img, lbl
                    
#                 # 生成粘贴位置
#                 y, x = np.where(bg_mask)
#                 paste_pos = random.choice(range(len(y)))
#                 y_start, x_start = y[paste_pos], x[paste_pos]
                
#                 # 获取对象区域
#                 y_obj, x_obj = np.where(mask)
#                 y_min, y_max = np.min(y_obj), np.max(y_obj)
#                 x_min, x_max = np.min(x_obj), np.max(x_obj)
#                 obj_h = y_max - y_min + 1
#                 obj_w = x_max - x_min + 1
                
#                 # 调整粘贴位置
#                 y_start = max(0, min(y_start, img.shape[0] - obj_h))
#                 x_start = max(0, min(x_start, img.shape[1] - obj_w))
                
#                 # 执行粘贴
#                 alpha = mask[y_min:y_max+1, x_min:x_max+1]
#                 img_patch = img[y_min:y_max+1, x_min:x_max+1]
#                 lbl_patch = lbl[y_min:y_max+1, x_min:x_max+1]
                
#                 img[y_start:y_start+obj_h, x_start:x_start+obj_w] = \
#                     img[y_start:y_start+obj_h, x_start:x_start+obj_w] * (1 - alpha) + img_patch * alpha
#                 lbl[y_start:y_start+obj_h, x_start:x_start+obj_w] = \
#                     lbl[y_start:y_start+obj_h, x_start:x_start+obj_w] * (1 - alpha) + lbl_patch * alpha
                    
#         return img, lbl
    
#     def intensity_aug(self, img):
#         """医学影像强度增强"""
#         # 保留原始值范围（假设输入是0-255）
#         orig_min, orig_max = np.min(img), np.max(img)
        
#         if random.random() < 0.5:
#             # 自适应直方图均衡化
#             clahe = cv2.createCLAHE(clipLimit=random.uniform(1.0,3.0), 
#                                   tileGridSize=(random.randint(4,16),)*2)
#             img = clahe.apply(img.astype(np.uint8)).astype(float)
            
#         if random.random() < 0.3:
#             # 局部对比度增强
#             img = cv2.addWeighted(img, random.uniform(1.2,1.5), 
#                                  img, 0, 
#                                  random.uniform(-10,10))
            
#         if random.random() < 0.3:
#             # 高斯噪声
#             noise = np.random.normal(0, random.uniform(0.5,2.0), img.shape)
#             img = np.clip(img + noise, orig_min, orig_max)
            
#         return img
    
#     def final_resize(self, img, lbl):
#         """最终尺寸标准化"""
#         h, w = self.output_size
#         if img.shape != (h, w):
#             img = zoom(img, (h/img.shape[0], w/img.shape[1]), order=3)
#             lbl = zoom(lbl, (h/lbl.shape[0], w/lbl.shape[1]), order=0)
#         return img, lbl

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# class MSCMRDataSets(Dataset):
#     def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.sup_type = sup_type
#         self.transform = transform
#         train_ids, test_ids = self._get_fold_ids(fold)

#         if self.split == 'train':
#             self.all_slices = os.listdir(self._base_dir + "/MSCMR_training_slices")
#             self.sample_list = []
#             for ids in train_ids:
#                 new_data_list = list(filter(lambda x: re.match(
#                     '{}.*'.format(ids), x) != None, self.all_slices))
#                 self.sample_list.extend(new_data_list)

#         elif self.split == 'val':
#             self.all_volumes = os.listdir(self._base_dir + "/MSCMR_training_volumes")
#             self.sample_list = []
#             for ids in test_ids:
#                 new_data_list = list(filter(lambda x: re.match(
#                     '{}.*'.format(ids), x) != None, self.all_volumes))
#                 self.sample_list.extend(new_data_list)

#         # if num is not None and self.split == "train":
#         #     self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))

#     def _get_fold_ids(self, fold):
#         training_set = ["subject{:0>2}".format(i) for i in
#                         [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
#         # validation_set = ["subject{:0>2}".format(i) for i in
#         #                   [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7,
#         #                    9]]
#         validation_set = ["subject{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
#         return [training_set, validation_set]

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train":
#             h5f = h5py.File(self._base_dir +
#                             "/MSCMR_training_slices/{}".format(case), 'r')
#         else:
#             h5f = h5py.File(self._base_dir +
#                             "/MSCMR_training_volumes/{}".format(case), 'r')
#         image = h5f['image'][:]
#         label = h5f['scribble'][:]
#         sample = {'image': image, 'label': label}

#         if self.split == "train":
#             image = h5f['image'][:]
#             if self.sup_type == "random_walker":
#                 label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
#             else:
#                 label = h5f[self.sup_type][:]
#             sample = {'image': image, 'label': label}
#             sample = self.transform(sample)


#         else:
#             image = h5f['image'][:]
#             # label = h5f[self.sup_type][:]
#             label = h5f['scribble'][:]
#             image=np.float64(image)
#             label=np.float64(label)
#             sample = {'image': image, 'label': label}

#         sample["idx"] = idx

#         return sample


# class ChaosDataset(Dataset):
#     def __init__(self, data_root, split, transform=None):
#         self.data_root = data_root
#         self.splits = split
#         self.transform = transform
#         self.train_ids = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34]
#         self.test_ids = [36, 37, 38, 39]

#         self.ids = self.train_ids if split == 'train' else self.test_ids
#         self.samples = []
#         for data_id in self.ids:
#             sample_path = os.path.join(data_root, str(data_id), 'T1DUAL/DICOM_anon/InPhase')

#             scribble_dir = os.path.join(data_root, str(data_id), 'T1DUAL/Ground_scribble')

#             input_files = os.listdir(sample_path)
#             label_files = os.listdir(scribble_dir)
#             input_files.sort()
#             label_files.sort()
#             assert len(input_files) == len(label_files), '输入和标签数量不一致'
#             for i, input_file in enumerate(input_files):
#                 input_path = os.path.join(sample_path, input_file)
#                 label_path = os.path.join(scribble_dir, label_files[i])
#                 self.samples.append((input_path, label_path))
#         print('数据集大小:', len(self.samples))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         input_path, label_path = self.samples[idx]
#         # 输入文件是dicom格式
#         input_img = pydicom.dcmread(input_path).pixel_array
#         label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#         # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
#         label_img = np.where(label_img == 0, 4, label_img)
#         label_img = np.where(label_img == 63, 0, label_img)
#         label_img = np.where(label_img == 126, 1, label_img)
#         label_img = np.where(label_img == 189, 2, label_img)
#         label_img = np.where(label_img == 252, 3, label_img)

#         sample = {'image': input_img, 'label': label_img}
#         if self.transform:
#             sample = self.transform(sample)

#         sample['idx'] = idx
#         return sample


class ChaosDataset(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",W=256,H=256):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.W = W
        self.H = H
        train_ids, test_ids = self._get_fold_ids(fold)
        self.all_slices = os.listdir(
            self._base_dir + "/MR")
        self.sample_list = []
        if self.split == 'train':
            for ids in train_ids:
                new_data_path = os.path.join(self._base_dir, "MR", str(ids), 'T1DUAL/DICOM_anon/InPhase')
                new_data_list = os.listdir(new_data_path)
                new_data_list = [os.path.join(new_data_path, i) for i in new_data_list if '.dcm' in i]
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            for ids in test_ids:
                new_data_path = os.path.join(self._base_dir, "MR",str(ids), 'T1DUAL/DICOM_anon/InPhase')
                new_data_list = os.listdir(new_data_path)
                new_data_list = [os.path.join(new_data_path, i) for i in new_data_list if '.dcm' in i]
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_ids = [1,2,3,5,8,13,15,19,20,21,22,31,32,33,34,36,37,38,39]
        all_cases_set = [i for i in all_ids]
        fold1_testing_set = [
            i for i in [1,2,3,5]]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            i for i in [8,13,15]]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            i for i in [19,20,21,22]]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            i for i in [31,32,33 ,34]]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            i for i in [36,37,38,39]]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = pydicom.dcmread(case).pixel_array
            label_path = case.replace('DICOM_anon/InPhase', 'Ground_scribble').replace('.dcm', '_scribble.png')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
            label = np.where(label == 0, 4, label)
            label = np.where(label == 63, 0, label)
            label = np.where(label == 126, 1, label)
            label = np.where(label == 189, 2, label)
            label = np.where(label == 252, 3, label)
        else:
            image = pydicom.dcmread(case).pixel_array
            image = image.astype(np.float32)
            label_path = case.replace('DICOM_anon/InPhase', 'Ground').replace('.dcm', '.png')
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
            label = np.where(label == 0, 4, label)
            label = np.where(label == 63, 0, label)
            label = np.where(label == 126, 1, label)
            label = np.where(label == 189, 2, label)
            label = np.where(label == 252, 3, label)
            
            #将测试集的图片resize到W*H
            image = zoom(image, (self.W / image.shape[0], self.H / image.shape[1]), order=0)
            label = zoom(label, (self.W / label.shape[0], self.H / label.shape[1]), order=0)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample["idx"] = idx
        sample['filename'] = case
        return sample

class MSCMRDataset(Dataset):
    def __init__(self, data_root, split, transform=None,W=256,H=256):
        self.W = W
        self.H = H
        self.data_root = data_root
        self.splits = split
        self.transform = transform
        if split == 'train':
            self.splits_dir = os.path.join(data_root, 'train')
        else:
            self.splits_dir = os.path.join(data_root, 'TestSet')

        self.samples = []
        self.files = []
        images_list = os.listdir(os.path.join(self.splits_dir, 'images'))
        labels_list = os.listdir(os.path.join(self.splits_dir, 'labels'))
        images_list.sort()
        labels_list.sort()
        assert len(images_list) == len(labels_list), '输入和标签数量不一致'
        for i, image_file in enumerate(images_list):
            image_path = os.path.join(self.splits_dir, 'images', images_list[i])
            label_path = os.path.join(self.splits_dir, 'labels', labels_list[i])
            img = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            for j in range(img.shape[2]):
                self.samples.append({'image': img[:, :, j], 'label': label[:, :, j]})
                self.files.append({'image': image_path})
        if split == 'train':
            self.splits_dir = os.path.join(data_root, 'val')
            images_list = os.listdir(os.path.join(self.splits_dir, 'images'))
            labels_list = os.listdir(os.path.join(self.splits_dir, 'labels'))
            images_list.sort()
            labels_list.sort()
            for i, image_file in enumerate(images_list):
                image_path = os.path.join(self.splits_dir, 'images', images_list[i])
                label_path = os.path.join(self.splits_dir, 'labels', labels_list[i])
                img = nib.load(image_path).get_fdata()
                label = nib.load(label_path).get_fdata()
                for j in range(img.shape[2]):
                    self.samples.append({'image': img[:, :, j], 'label': label[:, :, j]})
                    self.files.append({'image': image_path})
        

        print('数据集大小:', len(self.samples))

    def apply_clahe(self,img):
        """固定应用CLAHE增强"""
        if img.max() > 1:  # 兼容0-255和0-1两种范围
            img_norm = img.astype(np.uint8)
        else:
            img_norm = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img_norm).astype(np.float32) / 255.0
    
    def apply_gamma(self,img, gamma=0.7):
        """固定Gamma校正"""
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-7)
        return np.power(img_norm, gamma) * (img_max - img_min) + img_min
    
    def apply_homomorphic(self,img, cutoff=32, gamma_h=1.5, gamma_l=0.5):
        """固定频域增强"""
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        img_log = np.log1p(img.astype(np.float32))
        dft = np.fft.fft2(img_log)
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        x = np.linspace(-0.5, 0.5, cols)
        y = np.linspace(-0.5, 0.5, rows)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x**2 + y**2)
        h = (gamma_h - gamma_l) * (1 - np.exp(-(d**2 / (2*(cutoff**2))))) + gamma_l
        filtered_dft = dft * np.fft.fftshift(h)
        img_filtered = np.fft.ifft2(filtered_dft).real
        return np.exp(img_filtered) - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_img = self.samples[idx]['image']
        label_img = self.samples[idx]['label']
        input_img = self.apply_clahe(input_img)
        input_img = self.apply_gamma(input_img)
        input_img = self.apply_homomorphic(input_img)

        # #经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
        # label_img = np.where(label_img == 0, 4, label_img)
        # label_img = np.where(label_img == 63, 0, label_img)
        # label_img = np.where(label_img == 126, 1, label_img)
        # label_img = np.where(label_img == 189, 2, label_img)
        # label_img = np.where(label_img == 252, 3, label_img)
        #将测试集的图片resize到W*H
        input_img = zoom(input_img, (self.W / input_img.shape[0], self.H / input_img.shape[1]), order=0)
        label_img = zoom(label_img, (self.W / label_img.shape[0], self.H / label_img.shape[1]), order=0)
        sample = {'image': input_img, 'label': label_img}
        if self.transform:
            sample = self.transform(sample)

        sample['idx'] = idx
        sample['filename'] = self.files[idx]['image']
        return sample
# class MSCMRDataset(Dataset):
#     def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",W=256,H=256):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.sup_type = sup_type
#         self.transform = transform
#         self.W = W
#         self.H = H
        
#         if self.split == 'train':
#             # 遍历所有文件
#             train_list = []
#             for root, dirs, files in os.walk(self._base_dir + "/image/train"):
#                 for file in files:
#                     train_list.append(os.path.join(root, file))
#             val_list = []
#             for root, dirs, files in os.walk(self._base_dir + "/image/val"):
#                 for file in files:
#                     val_list.append(os.path.join(root, file))
            
#             self.sample_list += train_list
#             self.sample_list += val_list
#         else:
#             test_list = []
#             for root, dirs, files in os.walk(self._base_dir + "/image/test"):
#                 for file in files:
#                     test_list.append(os.path.join(root, file))
#             self.sample_list += test_list
          

#         print("total {} samples".format(len(self.sample_list)))


#     def __len__(self):
#         return len(self.sample_list)
    
#     def apply_clahe(self,img):
#         """固定应用CLAHE增强"""
#         if img.max() > 1:  # 兼容0-255和0-1两种范围
#             img_norm = img.astype(np.uint8)
#         else:
#             img_norm = (img * 255).astype(np.uint8)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         return clahe.apply(img_norm).astype(np.float32) / 255.0
    
#     def apply_gamma(self,img, gamma=0.7):
#         """固定Gamma校正"""
#         img_min, img_max = img.min(), img.max()
#         img_norm = (img - img_min) / (img_max - img_min + 1e-7)
#         return np.power(img_norm, gamma) * (img_max - img_min) + img_min
    
#     def apply_homomorphic(self,img, cutoff=32, gamma_h=1.5, gamma_l=0.5):
#         """固定频域增强"""
#         if img.max() <= 1:
#             img = (img * 255).astype(np.uint8)
#         img_log = np.log1p(img.astype(np.float32))
#         dft = np.fft.fft2(img_log)
#         rows, cols = img.shape
#         crow, ccol = rows//2, cols//2
#         x = np.linspace(-0.5, 0.5, cols)
#         y = np.linspace(-0.5, 0.5, rows)
#         x, y = np.meshgrid(x, y)
#         d = np.sqrt(x**2 + y**2)
#         h = (gamma_h - gamma_l) * (1 - np.exp(-(d**2 / (2*(cutoff**2))))) + gamma_l
#         filtered_dft = dft * np.fft.fftshift(h)
#         img_filtered = np.fft.ifft2(filtered_dft).real
#         return np.exp(img_filtered) - 1

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         image = cv2.imread(case, cv2.IMREAD_GRAYSCALE)
#         image = self.apply_clahe(image)
#         image = self.apply_gamma(image)
#         image = self.apply_homomorphic(image)
#         label_path = case.replace('image', 'scribble').replace('DE_img', 'DE_scribble').replace('.png', '_scribble.png')
#         if self.split == "train":
           
#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#             # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
#             label = np.where(label == 0, 4, label)
#             label = np.where(label == 252, 0, label)
#             label = np.where(label == 63, 1, label)
#             label = np.where(label == 189, 2, label)
#             label = np.where(label == 126, 3, label)

#         else:
            

#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#             # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
#             label = np.where(label == 0, 4, label)
#             label = np.where(label == 252, 0, label)
#             label = np.where(label == 63, 1, label)
#             label = np.where(label == 189, 2, label)
#             label = np.where(label == 126, 3, label)
#             #将测试集的图片resize到W*H
#             image = zoom(image, (self.W / image.shape[0], self.H / image.shape[1]), order=0)
#             label = zoom(label, (self.W / label.shape[0], self.H / label.shape[1]), order=0)


#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
            
#         sample["idx"] = idx
#         sample['filename'] = case
#         return sample