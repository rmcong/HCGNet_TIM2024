from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_RGBDD import get_eval_set
from functools import reduce
import scipy.io as sio
import time
from imageio import imwrite
import cv2
import numpy as np
from PIL import Image

from HCGNet import Net as HCGNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
# parser.add_argument('--input_dir', type=str, default='./data/')
# parser.add_argument('--bit16_dir', type=str, default='test_x8/')
# parser.add_argument('--output', default='./results/x8_Lu',
#                     help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='DepthLr/')
parser.add_argument('--test_rgb_dataset', type=str, default='RGB/')
parser.add_argument('--test_gt_dataset', type=str, default='DepthHR/')
parser.add_argument('--model_type', type=str, default='HCGNet')
parser.add_argument('--model',
                    default="./weights_tim/nyu_x16.pth", help='sr pretrained base model')
opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)


def eval(opt):
    rmse_list = []
    total_time = []

    for dataset_name in ['Lu', 'RGBDD']:
        if dataset_name == 'Lu':
            data_path = './data/RGBD_test/test_AfterProcessing_' + str(opt.upscale_factor) + 'X' + '/Lu/'
            save_path = opt.output + '/Lu'
        elif dataset_name == 'RGBDD':
            data_path = './data/RGBD_test/test_AfterProcessing_' + str(opt.upscale_factor) + 'X' + '/RGBDD/'
            save_path = opt.output + '/RGBDD'

        print("===> Loading dataset %s" % (dataset_name))
        test_set = get_eval_set(os.path.join(data_path, opt.test_gt_dataset),
                                os.path.join(data_path, opt.test_dataset),
                                os.path.join(data_path, opt.test_rgb_dataset)
                                )
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                         shuffle=False)
 
        print('===> Building model')
        if opt.model_type == 'HCGNet':
            model = HCGNet(num_channels=1, base_filter=64, feat=256, num_stages=3,
                               scale_factor=opt.upscale_factor)  

        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if cuda:
            torch.cuda.manual_seed(opt.seed)
        if cuda:
            model = model.cuda(gpus_list[0])
 
        if os.path.exists(opt.model):
            model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.<---------------------------->')

        model.eval()
        torch.set_grad_enabled(False)
        for batch in testing_data_loader:
            input, input_rgb, target, name = Variable(batch[0]), Variable(batch[1]), \
                                                           batch[2], batch[3],

            if opt.gpu_mode:
                input = input.cuda(gpus_list[0])
                input_rgb = input_rgb.cuda(gpus_list[0])
                target = target.cuda(gpus_list[0])

            t0 = time.time()
            prediction = model(input_rgb, input)
            t1 = time.time()
            total_time.append(t1-t0)
            # print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            # print(prediction.cpu().data.shape)

            gt = target.cpu().data.squeeze().numpy()
            D_min = gt.min()
            D_max = gt.max()
            img = prediction.cpu().data.squeeze().clamp(0, 1).numpy()
            img = (img * (D_max - D_min)) + D_min

            if  dataset_name == 'Lu':
                img_com = img.clip(min=0, max=255)
                gt_com = gt.clip(min=0, max=255)
                rmse = Rmse(img_com, gt_com, rmse_list)
                # save_img(img, name[0], save_path)
            else:
                img = img * 100
                gt = gt * 100
                rmse = Rmse(img, gt, rmse_list)
                # save_16bit_img(img * 10, name[0], save_path)

        rmse_list = []
        del(total_time[0])
        print("%s meanRMSE:" % (dataset_name), np.mean(rmse))
        print("Avg Time: %.4f sec." % np.mean(total_time))
        total_time = []


def save_img(img, img_name, save_path):
    # save_img = img.squeeze().clamp(0, 1).numpy()

    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _, index = os.path.split(img_name)
    save_fn = save_dir + '/' + index + '.png'

    cv2.imwrite(save_fn, img)
    # imwrite(save_fn, save_img)


def save_16bit_img(img, img_name, save_path):

    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _, index = os.path.split(img_name)
    save_fn = save_dir + '/' + index + '.png'

    cv2.imwrite(save_fn, img.astype('uint16'))
    # imwrite(save_fn, save_img)


def Rmse(img, gt, rmse_list):
    # img1, img2: [0, 255]
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"

    diff = gt - img
    h, w = diff.shape
    rmse = np.sqrt(np.sum(np.power(diff, 2) / (h * w)))
    rmse_list.append(rmse)
    return rmse_list

##Eval Start!!!
if __name__ == '__main__':
    eval(opt)
