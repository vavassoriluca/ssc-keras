"""
SSC300 implementation based on the Keras port of the original Caffe SSD300 network by Pierluigi Ferrari.
"""

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Concatenate, LSTM, \
    Bidirectional, Reshape, Activation, BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from object_counting.keras_ssc.keras_layers.keras_layer_L2Normalization import L2Normalization


def ssc_300(image_size,
            n_classes,
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            predictors=['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2'],
            hidden_size=[250, 250, 100],
            output_activation=False,
            lstm=False,
            condense_predictors=False):
    """
    Build a Keras model with SSC300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper. Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    References: https://arxiv.org/abs/1512.02325v5

    :param tuple image_size: The input image size in the format `(height, width, channels)`.
    :param int n_classes: The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
    :param float l2_regularization: The L2-regularization rate. Applies to all convolutional layers.
        Set to zero to deactivate L2-regularization.
    :param float min_scale: The smallest scaling factor for the size of the anchor boxes as a fraction
        of the shorter side of the input images.
    :param float max_scale: The largest scaling factor for the size of the anchor boxes as a fraction
        of the shorter side of the input images. All scaling factors between the smallest and the
        largest will be linearly interpolated. Note that the second to last of the linearly interpolated
        scaling factors will actually be the scaling factor for the last predictor layer, while the last
        scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
        if `two_boxes_for_ar1` is `True`.
    :param list scales: A list of floats containing scaling factors per convolutional predictor layer.
        This list must be one element longer than the number of predictor layers. The first `k` elements are the
        scaling factors for the `k` predictor layers, while the last element is used for the second box
        for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
        last scaling factor must be passed either way, even if it is not being used. If a list is passed,
        this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
    :param list aspect_ratios_global: The list of aspect ratios for which anchor boxes are to be
        generated. This list is valid for all prediction layers.
    :param list aspect_ratios_per_layer: A list containing one aspect ratio list for each prediction layer.
        This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
        original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
    :param bool two_boxes_for_ar1: Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
        If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
        using the scaling factor for the respective layer, the second one will be generated using
        geometric mean of said scaling factor and next bigger scaling factor.
    :param list steps: `None` or a list with as many elements as there are predictor layers. The elements can be
        either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
        pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
        the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
        If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
        If no steps are provided, then they will be computed such that the anchor box center points will form an
        equidistant grid within the image dimensions.
    :param list offsets: `None` or a list with as many elements as there are predictor layers. The elements can be
        either floats or tuples of two floats. These numbers represent for each predictor layer how many
        pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
        as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
        of the step size specified in the `steps` argument. If the list contains floats, then that value will
        be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
        `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
    :param list subtract_mean: `None` or an array-like object of integers or floating point values
        of any shape that is broadcast-compatible with the image shape. The elements of this array will be
        subtracted from the image pixel intensity values. For example, pass a list of three integers
        to perform per-channel mean normalization for color images.
    :param list divide_by_stddev: `None` or an array-like object of non-zero integers or
        floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
        intensity values will be divided by the elements of this array. For example, pass a list
        of three integers to perform per-channel standard deviation normalization for color images.
    :param list swap_channels: Either `False` or a list of integers representing the desired order in which the input
        image channels should be swapped.
    :param list predictors: names of the convolutional layers used as predictors
    :param list hidden_size: number of neurons for the 3 hidden fully-connected layers
    :param bool output_activation: whether to include or not the softplus activation function after the hidden layers
    :param bool lstm: whether to add or not an LSTM cell on top of the hidden layer
    :param bool condense_predictors: whether to condense or not the predictors in a single prediction

    :return model: The Keras SSC300 model.
    """

    n_predictor_layers = len(predictors)  # The number of predictor conv layers in the network is 6 for the original SSD300.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(hidden_size) != 3:
        raise ValueError("3 hidden size values must be passed, but {} values were received.".format(len(hidden_size)))
    hidden_size= np.array(hidden_size)
    if np.any(hidden_size <= 0):
        raise ValueError("All hidden sizes must be >0, but the sizes given are {}".format(hidden_size))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    conv_features = {'conv4_3': conv4_3_norm,
                     'fc7': fc7,
                     'conv6_2': conv6_2,
                     'conv7_2': conv7_2,
                     'conv8_2': conv8_2,
                     'conv9_2': conv9_2}
    predictor_layers = []

    ### Build the predictor layers on top of the base network
    for predictor in predictors:
        flatten = Flatten(name='{}_flat'.format(predictor))(conv_features[predictor])
        d1 = Dense(hidden_size[0], name='{}_d1'.format(predictor))(flatten)
        d1bn = BatchNormalization(name='{}_bn1'.format(predictor))(d1)
        r1 = Activation(activation='relu', name='{}_r1'.format(predictor))(d1bn)
        d2 = Dense(hidden_size[1], name='{}_d2'.format(predictor))(r1)
        d2bn = BatchNormalization(name='{}_bn2'.format(predictor))(d2)
        r2 = Activation(activation='relu', name='{}_r2'.format(predictor))(d2bn)
        d3 = Dense(hidden_size[2], name='{}_d3'.format(predictor))(r2)
        d3bn = BatchNormalization(name='{}_bn3'.format(predictor))(d3)
        r3 = Activation(activation='relu', name='{}_r3'.format(predictor))(d3bn)
        pred = Dense(n_classes, name='{}_pred'.format(predictor))(r3)
        predictor_layers.append(pred)

    # Concatenate the output of the different predictors
    # Output shape of `predictions`: (batch, n_predictors, n_classes)
    predictions = Concatenate(axis=1, name='predictions1')(predictor_layers)
    if output_activation:
        predictions = Activation(activation='softplus')(predictions)
    if lstm:
        predictions = Reshape((n_predictor_layers, n_classes), name='lstm_predictions_res')(predictions)
        predictions = Bidirectional(LSTM(20, return_sequences=False), name='lstm_predictions')(predictions)
    if condense_predictors:
        predictions = Dense(n_classes, name='predictions_condensed')(predictions)

    return Model(inputs=x, outputs=predictions)
