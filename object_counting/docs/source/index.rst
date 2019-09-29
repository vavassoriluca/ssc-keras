.. Counting Objects with Deep Learning documentation master file, created by
   sphinx-quickstart on Tue May  7 10:49:58 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###############################################################
Welcome to Counting Objects with Deep Learning's documentation!
###############################################################

This is an API that provides different models to count objects in images and train them on customized datasets.
It uses **Python 3** and **Keras** with **Tensorflow** backend. The models are implementations or improvements of the techniques proposed in Chattopadhyay et al. (2017)  `Counting Everyday Objects in Everyday Scenes`_ and the novel **SSC Single-Shot Multiscale Counter** proposed during the internship of Vavassori Luca at KPMG Nederland.

.. _Counting Everyday Objects in Everyday Scenes: https://arxiv.org/abs/1604.03505

******************
How to use the API
******************
In the main project folder there are several notebooks with a detailed explanation of all the steps necessary in order to use the API,
load and sample weights and train the models on custom datasets. More in detail, the notebooks are:

1. **CountingObjectsEverydayLiteratureModels**: Implementation of the algorithm from `Counting Everyday Objects in Everyday Scenes`_
2. **SSC300**: Usage of the SSC300 model.
3. **SSC300 Evaluation**: Evauation of the variants of SSC300.
4. **Spread count of BBoxes**: Bivariate Normal Distribution to split the count of objects in a grid using a bivariate gaussian.

Project Requirements
====================

This project relies on the following main libraries and frameworks:

- Python 3.7
- Keras 2.2.4
- Tensorflow 1.13.1
- numpy
- torchfile
- tqdm
- h5py

A requirement file to set up a virtual environment is provided as well::

   pip install -r requirements.txt


******************
Models explanation
******************

.. toctree::
   :maxdepth: 2

   models_expl

*************
Documentation
*************

.. toctree::
   :maxdepth: 2

   ssc
   obj_models
   convert_gt
   custom_layers
   utils

*************************
Documentation in one page
*************************

.. toctree::
   :maxdepth: 2

   api

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
