#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/27

import sys
import numpy as np
import Image
sys.path.insert(0, "../")
import tools
tools.load_caffe()
import caffe

img_input_path = '../images/lena_gray.png'
img_output_path = 'lena_out.png'
net_definition = 'net.prototxt'

caffe.set_mode_gpu()
net = caffe.Net(net_definition, caffe.TEST)

img = np.array(Image.open(img_input_path))

# preprocess image
img_input = img[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*img_input.shape)
net.blobs['data'].data[...] = img_input

# apply filter
net.forward()

img_output = tools.normalize_to_range(net.blobs['pool'].data[0, 0], 255)
tools.save_grayscale_image_from_arary(img_output, img_output_path)
