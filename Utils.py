# from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
# from numpy.random import randint
from scipy.misc import imresize
import os
import sys
import imageio
from glob import glob

def dummy_load_training_data (batch_size=1, is_testing=False, img_res=(256, 256)):
    
    data_type = "train" if not is_testing else "test"
        
    path = glob('./datasets/%s/*' % ('img_align_celeba'))

    batch_images = np.random.choice(path, size=batch_size)

    imgs_hr = []
    imgs_lr = []
    for img_path in batch_images:
        img = imread(img_path)

        h, w = img_res
        low_h, low_w = int(h / 4), int(w / 4)

        img_hr = imresize(img, img_res)
        img_lr = imresize(img, (low_h, low_w))

        # If training => do random flip
        if not is_testing and np.random.random() < 0.5:
            img_hr = np.fliplr(img_hr)
            img_lr = np.fliplr(img_lr)

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    return imgs_hr, imgs_lr

def imread(path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)