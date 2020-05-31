# Image Analysis with Deep Learning: CNN, Object Detection and Object Counting through Keras

In this repository you can find the work I have done during my master thesis project. The main projects I have developed are related to Object Detection and Object Counting. 

The code has been written in Python 3.6 and takes advantage of the Keras deep learnign library based on Tensorflow backed (no other backends available at the moment). 

## Packages
This repository provides 3 main modules:
1. **object_classification**: provides a class to instanciate different classification models implemented in Keras
2. **object_counting**: it consists of 2 main submodules that are:
	- *objects_counter*:  implementation of the models described in [Counting Everyday Objects in Everyday Scenes](https://arxiv.org/abs/1604.03505), Chattopadhyay et al, 2017.
	- *keras_ssc*: implementation of the new SSC model proposed during the thesis.
3. **object_detection**: it provides the implementation of the Keras SSD Object Detector by [Pierluig Ferrari](https://github.com/pierluigiferrari/ssd_keras).

