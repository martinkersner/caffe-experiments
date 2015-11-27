# Caffe experiments

Experiments with deep learning framework [Caffe](http://caffe.berkeleyvision.org/).
All experiments are going to be performed on image, particularly using [Lena](https://en.wikipedia.org/wiki/Lenna) grayscale or RGB image.
Size of both images is 512 x 512 px, however in this README images are displayed twice smaller.

<p align="center">
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/images/lena_gray.png" height=256 width=256 title="Lena grayscale"/>
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/images/lena.png" height=256 width=256  title="Lena RGB"/> 
</p>

## Gaussian filter
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/gaussian_filter/net.png" title="Gaussian filter net"/>

Output of this layer depends on gaussian initialization inside of *weight_filler*. Different initialization produces different output images.

<p align="center">
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/gaussian_filter/lena_out_0.png" height=256 width=256 title="Output of Gaussian filter net"/>
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/gaussian_filter/lena_out_1.png" height=256 width=256 title="Output of Gaussian filter net"/>
</p>

## Pooling layer
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/pooling_layer/net.png" title="Pooling layer net"/>

Output image of pooling layer is twice smaller because of parameter *stride* 2.

<p align="center">
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/pooling_layer/lena_out.png" height=128 width=128 title="Output of pooling layer with parameters kernel_size: 2 and stride: 2"/>
</p>

When we change parameter *kernel_size* to 10 and *stride* to 1, output image becomes blurry, because max values are computed from larger region while retaining input size of image (only borders are cropped a bit). Output image size is 503 x 503 px.

<p align="center">
<img src="https://raw.githubusercontent.com/martinkersner/caffe-experiments/master/pooling_layer/lena_10_1.png" height=256 width=256 title="Output of pooling layer with parameters kernel_size: 10 and stride: 1"/>
</p>

Pooling layer isn't dependant on any random initialization, therefore output images from pooling layer with the same parameters will be always the same.
