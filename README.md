# Image Analysis with Deep Learning: CNN, Object Detection and Object Counting through Keras

In this repository you can find the work I have done during my master thesis internship at KPMG as an intern of the Advanced Analytics and Big Data department. The main projects I have developed are related to Object Detection and Object Counting. 

The code has been written in Python 3.6 and takes advantage of the Keras deep learnign library based on Tensorflow backed (no other backends available at the moment). 

## Packages
This repository provides 3 main modules:
1. **object_classification**: provides a class to instanciate different classification models implemented in Keras
2. **object_counting**: it consists of 2 main submodules that are:
	- *objects_counter*:  implementation of the models found in the literature during the thesis.
	- *keras_ssc*: implementation of the new SSC model proposed during the thesis.
3. **object_detection**: it provides the implementation of the SSD detector in Keras.

## CNNs
CNN is a deep learning model whose architecture is able to extract features out of images and different domains data encoded as an image. Those features can be used to perform different tasks, the most common one is Image Classification. Other interesting use cases are Image Recognition (Localisation, Detection), Scene Understanding, Object Counting and Image Segmentation. 

CNNs learn weights in the form of kernels/filters, which can be seen as squared masks that multiply a 3-dimensional tensor sliding on it, exploting the spatial information of the data.

![Convolution Example](https://mlnotebook.github.io/img/CNN/convSobel.gif)

*Example of convolutional operation using one 3x3 kernel with stride 1*

### Convolution Operation
In a convolutional layer, the main parameter to be defined are:
- number of kernels: the number of filters the convolutional layer will learn and apply to the input
- size of the kernel: the size of the filter (always square-shaped)
- stride: the number of cells to be jump while sliding (the value is adopted both for the right and down shift)
- padding: whether to add a zero-padding around the image to preserve the same width and height between input and output
- dilation: technique used to increase the receptive field of the kernel

For a better understanding of the basic of CNN have a look at [this blog post](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/). 


