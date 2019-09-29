###################################
SSC: Single-Shot Multiscale Counter
###################################

SSC is an end-to-end mode able to count objects of different classes in picture of size 300x300 with an evaluated mRMSE of 0.35 on the PASCAL VOC 2007 test set.
It consists of the architecture of the SSD detector with a modification in the last layers to count. How to use the model and its variants is explained in the following notebooks:

1. SSC300.ipynb
2. SSC300 Evaluation.ipynb

The original implementation of the SSD detector keras port is from Pierluigi Ferrari and it available at this link: `SSD Detector <https://github.com/pierluigiferrari/ssd_keras>`_
There are up to 6 predictors for 6 different scales (the optimal predictors are the topmost 4 predictors of SSD).
It is possible to obtain the prediction either split per scale or reduced to a single prediction according to the variant.

The following modules contribute to the package

.. toctree::
    :maxdepth: 2

    ssc_models
    ssc_bboxes
    ssc_layers
    ssc_encoder
    ssc_data


