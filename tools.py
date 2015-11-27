#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/20

import os
import sys
import Image
import numpy as np

def load_caffe():
    caffe_root = os.environ['CAFFE_ROOT']
    sys.path.insert(0, caffe_root + '/python')

def save_rgb_image_from_arary(img_arr, img_name):
    im = Image.fromarray(img_arr)
    im.convert('RGB').save(img_name)

def save_grayscale_image_from_arary(img_arr, img_name):
    img_arr = img_arr.astype("uint8")
    im = Image.fromarray(img_arr)
    im.save(img_name)

def normalize_to_range(img, max_thresh):
    img_min = np.min(img),
    img_max = np.max(img)
    diff = img_max - img_min

    return ((img - img_min) / diff) * max_thresh

def preprocess_image(net, img_path):
    img = np.array(Image.open(img_path))

    # preprocess image
    img_input = img[np.newaxis, np.newaxis, :, :]
    net.blobs['data'].reshape(*img_input.shape)
    net.blobs['data'].data[...] = img_input

def apply_filter(net, layer_name):
    net.forward()
    img_output = net.blobs[layer_name].data[0, 0]
    img_norm = normalize_to_range(img_output, 255)

    return img_norm
