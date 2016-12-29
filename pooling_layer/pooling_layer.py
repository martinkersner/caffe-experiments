#!/usr/bin/env python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/27

import sys

sys.path.insert(0, "../") # add tools.py
import paths as p
import tools as tl

tl.load_caffe()
import caffe

layer_name = "pool"

caffe.set_mode_gpu()
net = caffe.Net(p.net_definition, caffe.TEST)
tl.preprocess_image(net, p.img_input_path)
img_output = tl.apply_filter(net, layer_name)
tl.save_grayscale_image_from_arary(img_output, p.img_output_path)
