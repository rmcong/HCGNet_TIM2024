from PIL import Image
import numpy as np

from glob import glob
import math
import cv2
import os
from six.moves import xrange
from PIL import Image


dataset_pre = './results/x8_middle_TIM'
dataset_gt = './data/midde_test/test_gt'

data1 = sorted(glob(os.path.join(
                dataset_pre, "*.png")))
data2 = sorted(glob(os.path.join(
                dataset_gt, "*.png")))
mad_list = []

for filename in data1:
    for filename2 in data2:
    
        i = filename.split('/')[-1].split('.')[0]
        j = filename2.split('/')[-1].split('.')[0]

        if i == j:
            print(i)

            img_pre = cv2.imread(filename, 0)
            img_gt = cv2.imread(filename2, 0)

            img_pre = img_pre.astype(np.double)
            img_gt =img_gt.astype(np.double)

            diff = img_gt - img_pre

            h, w = diff.shape
            mad = np.sum(abs(diff)) / (h * w)

            mad_list.append(mad)



print('mad: ', mad_list)
print('mean:', np.mean(mad_list))
