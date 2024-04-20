from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set_middle
from functools import reduce
import scipy.io as sio
import time
from imageio import imwrite
import cv2
import numpy as np
from PIL import Image

from HCGNet import Net as HCGNet


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='./data/midde_test/')
parser.add_argument('--nyu_dir', type=str, default='test_x8/')
parser.add_argument('--output', default='results/x8_middle_TIM',
                    help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='test_x8/')
parser.add_argument('--test_rgb_dataset', type=str, default='test_color/')
parser.add_argument('--model_type', type=str, default='HCGNet')
parser.add_argument('--model',
                    default="./weights_tim/middle_x8.pth",
                    help='sr pretrained base model')

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set_middle(os.path.join(opt.input_dir, opt.test_dataset),
                        os.path.join(opt.input_dir, opt.test_rgb_dataset))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'HCGNet':
    model = HCGNet(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=opt.upscale_factor)  

if os.path.exists(opt.model):
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.<---------------------------->')

if cuda:
    model = model.cuda(gpus_list[0])

rmse_list = []
mad_list = []

def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        input, input_rgb, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        # print(input_rgb)
        if cuda:
            input = input.cuda(gpus_list[0])
            input_rgb = input_rgb.cuda(gpus_list[0])
        t0 = time.time()
        prediction = model(input_rgb, input)
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        # print(prediction.cpu().data.shape)
        
        save_img(prediction.cpu().data, name[0])


def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy()

    save_dir = opt.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, save_img * 255)
    # imwrite(save_fn, save_img)


def save_nyu_img(img, img_name):

    save_dir = os.path.join(opt.output, opt.nyu_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, img.astype('uint16'))
    # imwrite(save_fn, save_img)


##Eval Start!!!
if __name__ == '__main__':
    eval()
