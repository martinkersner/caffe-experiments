#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/20

import os
import sys
import Image
import numpy as np

def save_image_from_arary(img_arr, img_name):
    im = Image.fromarray(img_arr)
    im.convert('RGB').save(img_name)

def load_caffe():
    caffe_root = os.environ['CAFFE_ROOT']
    sys.path.insert(0, caffe_root + '/python')

def normalize_to_range(img, max_thresh):
    img_min = np.min(img),
    img_max = np.max(img)
    diff = img_max - img_min

    return ((img - img_min) / diff) * max_thresh
