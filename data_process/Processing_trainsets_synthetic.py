# -*- coding: utf-8 -*-

import os
import h5py
import torch

from glob import glob
import numpy as np
import torch.utils.data as Data

from scipy.io import loadmat
from skimage.color import rgb2ycbcr
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import random

import math
import cv2
import os
from PIL import Image
import time


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (c, ih, iw) = img_in.shape
    ####print('input:', ih, iw)
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[iy:iy + ip, ix:ix + ip]
    # print('get_patch', img_tar.size(), ty, ty + tp, tx, tx + tp)
    # img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]
    # info_patch = {
    # 	'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    ####print('after', img_tar.size())

    # return img_in, img_tar, info_patch
    return img_in,img_tar


def new_folder(result_root):
    if not os.path.exists(result_root):
        os.makedirs(result_root)


def processing_trainsets_synthetic(scale,data_path,save_path,dataset_name):

    Depth_files = sorted(get_img_file(os.path.join(data_path, 'depth_train')))
    RGB_files = sorted(get_img_file(os.path.join(data_path, 'RGB_train')))

    counter = 0
    for image_index in range(len(Depth_files)):
        image = np.transpose(imread(RGB_files[image_index]).astype(
            'float32'), [2, 0, 1]) # [3,h,w] 0~255
        if dataset_name == 'Middlebury' or dataset_name == 'Lu':
            depth_hr = imread(Depth_files[image_index]).astype(
                'float32')   # [h,w] 0~255
        elif dataset_name == 'NYU':
            depth_hr = np.load(Depth_files[image_index]).astype(
                'float32')
        elif dataset_name == 'RGBDD':
            depth_hr = imread(Depth_files[image_index]).astype(
                'float32') /1000  # [h,w] 0~3000+(mm)

        if scale == 4:
            patch_size = 64
            iteration = 20
        elif scale == 8:
            patch_size = 128
            iteration = 16
        elif scale == 16:
            patch_size = 256
            iteration = 16

        new_folder(os.path.join(save_path, 'DepthHR'))
        new_folder(os.path.join(save_path, 'RGB'))
        new_folder(os.path.join(save_path, 'DepthLr'))

        h, w = image.shape[1:]

        for index in tqdm(range(0, iteration)):
            color_patch, depth_patch = get_patch(image, depth_hr, patch_size, scale)
            depth_patch_lr = np.array(Image.fromarray(depth_patch).resize(
                (patch_size // scale, patch_size // scale), Image.BICUBIC))  # bicubic

            # save the patches in npy
            filename = os.path.splitext(Depth_files[image_index].split('\\')[-1])[0]
            np.save(os.path.join(save_path, 'DepthHR',
                                 str(counter) + '.npy'), depth_patch[None, :, :].astype(np.float32))
            np.save(os.path.join(save_path, 'RGB',
                                 str(counter) + '.npy'), color_patch.astype(np.float32))
            np.save(os.path.join(save_path, 'DepthLr',
                                 str(counter) + '.npy'), depth_patch_lr[None, :, :].astype(np.float32))

            counter += 1
    
        # # get LR Depth map
        # h, w = image.shape[1:]
        # depth_lr = np.array(Image.fromarray(depth_hr).resize(
        #     (w//scale, h//scale), Image.BICUBIC))  # bicubic
        #
        # # normalize depth map
        # depth_min = depth_hr.min()
        # depth_max = depth_hr.max()
        # assert depth_min != depth_max
        # # depth_hr_norm = (depth_hr - depth_min) / (depth_max - depth_min)
        # depth_lr_norm = (depth_lr - depth_min) / (depth_max - depth_min)
        #
        # # normalize RGB image
        # image = image.astype(np.float32)/255  # [3, H, W] 0~1
        # image_norm = (image - np.array([0.485, 0.456, 0.406]).reshape(3,
        #          1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        #
        # # follow DKN & FDSR use bicubic upsampling of PIL
        # depth_lr_up = np.array(Image.fromarray(
        #     depth_lr_norm).resize((w, h), Image.BICUBIC))
        #
        # new_folder(os.path.join(save_path, 'DepthHR'))
        # new_folder(os.path.join(save_path, 'RGB'))
        # new_folder(os.path.join(save_path, 'DepthLrUp'))
        #
        # # save the patches in npy
        # filename = os.path.splitext(Depth_files[image_index].split('\\')[-1])[0]
        # np.save(os.path.join(save_path, 'DepthHR',
        #         filename+'-depthHr.npy'), depth_hr[None, :, :].astype(np.float32))
        # np.save(os.path.join(save_path, 'RGB',
        #         filename+'-RGB.npy'), image_norm.astype(np.float32))
        # np.save(os.path.join(save_path, 'DepthLrUp',
        #         filename+'-depthLrUp.npy'), depth_lr_up[None, :, :].astype(np.float32))
