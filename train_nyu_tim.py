from __future__ import print_function
import argparse
from math import log10
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_RGBDD import get_training_set
import pdb
import socket
import time
import cv2

from HCGNet import Net as HCGNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--StartEpoch', type=int, default=60, help='The number of the start epoch')
parser.add_argument('--snapshots', type=int, default=30, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./data/NYUDepthv2/depth_train_8X/')  
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DepthHR/')
parser.add_argument('--rgb_train_dataset', type=str, default='RGB/')
parser.add_argument('--train_dataset', type=str, default='DepthLr/')
parser.add_argument('--model_type', type=str, default='HCGNet')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='./weights/', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='_srh', help='Location to save checkpoint models')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):

        input_rgb, input, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

        if cuda:
            input_rgb = input_rgb.cuda()
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input_rgb, input)
        # save_img(prediction.cpu().data, iteration)
        loss = criterion(prediction, target)

        t1 = time.time()

        if loss.item() > 10 or torch.isnan(torch.tensor(loss.item())):
            # print('--------------------------------------nan')
            continue

        epoch_loss += loss.item()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)

        optimizer.step()

        avg_loss = epoch_loss / len(training_data_loader)
        loss_list.append(avg_loss)

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.item(),
                                                                                 (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def new_folder(result_root):
    if not os.path.exists(result_root):
        os.makedirs(result_root)


def checkpoint(epoch):
    new_folder(opt.save_folder)
    model_out_path = opt.save_folder + hostname + opt.model_type + opt.prefix + "_epoch_{}.pth".format(
        epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.train_dataset, opt.hr_train_dataset, opt.rgb_train_dataset,
                             opt.upscale_factor, opt.patch_size, opt.data_augmentation)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'HCGNet':
    model = HCGNet(num_channels=1, base_filter=64, feat=256, num_stages=3, scale_factor=opt.upscale_factor)

# model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()
# criterion = nn.MSELoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.pretrained_sr)
    # model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    print(model_name)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('************************************Pre-trained SR model is loaded.************************************')
###############
if cuda:
    model = model.cuda()
    # model = nn.DataParallel(model)
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), eps=1e-8)

loss_list = []
for epoch in range(opt.StartEpoch, opt.nEpochs + 1):
    train(epoch)

    if (epoch + 1) % 10 == 0:
        plt.plot(loss_list, linewidth=5)
        plt.title('Loss Table', fontsize=24)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig('AFP_3times.png')

    if (epoch + 1) == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) == 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0 and epoch <= 100:
        checkpoint(epoch)
    elif (epoch + 1) == 1:
        checkpoint(epoch)
    elif (epoch + 1) == opt.nEpochs:
        checkpoint(epoch)
    elif epoch > 100 and (epoch + 1) % 10 == 0:
        checkpoint(epoch)
    # checkpoint(epoch)


