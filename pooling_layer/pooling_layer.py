#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/27

import sys
import numpy as np
import Image

sys.path.insert(0, "../") # add tools.py
import paths as p
import tools

tools.load_caffe()
import caffe

caffe.set_mode_gpu()
net = caffe.Net(p.net_definition, caffe.TEST)

img = np.array(Image.open(p.img_input_path))

# preprocess image
img_input = img[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*img_input.shape)
net.blobs['data'].data[...] = img_input

# apply filter
net.forward()

img_output = tools.normalize_to_range(net.blobs['pool'].data[0, 0], 255)
tools.save_grayscale_image_from_arary(img_output, p.img_output_path)
