from object_counting.objects_counter.utils.base_net_e2e import BaseNetE2E
from object_counting.objects_counter.models_e2e.counting_model_e2e import CountingModelE2E

from keras.models import Model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import ReLU

import numpy as np


class AsoSubE2E(CountingModelE2E):

    """The :class:`AsoSub` model performs associative subitizing count of an image. The output is for a single cell,
    so it must be summed up among all the cell of an image at the end of the prediction.

    The model consists of the same architecture of :class:`models.glance.Glance`, with the only difference that the input is an individual cell rather then the whole image.

    :param tuple input_shape: shape of the input
    :param str base_model: the name of the network used to extract features
    :param int num_classes: num of output classes
    :param num_hidden: number of hidden layers in the MLP
    :param hidden_size: number of activations in the hidden layers in the MLP
    :param include_relu: whether to include relu after the final fully-connected layer
    :param boolean freeze_base_net: whether to set non-trainable or trainable the layers of the base_net
    """

    def __init__(self, input_shape, base_model='resnet152', num_classes=20, num_hidden=1, hidden_size=500, include_relu=False, freeze_base_net=True, weights=None):

        super().__init__(input_shape, base_model, num_classes=num_classes)

        self.base_model, added_layers = BaseNetE2E(base_model, input_shape, conv_output=True).get_model()

        freeze_len = len(self.base_model.layers) - added_layers

        model = Dense(hidden_size)(self.base_model.layers[-1].output)
        model = BatchNormalization()(model)
        model = ReLU()(model)
        if num_hidden > 1:
            for i in range(num_hidden-1):
                model = Dense(hidden_size)(model)
                model = BatchNormalization()(model)
                model = ReLU()(model)
        model = Dense(self.num_classes)(model)
        if include_relu:
            model = ReLU()(model)

        self.model = Model(inputs=self.base_model.input, outputs=model)

        if freeze_base_net:
            for l in self.model.layers[:freeze_len]:
                l.trainable = False

    def predict(self,
                X,
                batch_size=None,
                steps=None):

        """This method overrides the :meth:`CountingModelE2E.predict`. It perform a normal prediction and finally decode the output to compute the global count

        :param np.ndarray X: samples, first dimension number of samples, second dimension number of cells
        :param int batch_size: size of a batch
        :param int steps: number of batches to process
        """

        predictions = super().predict(X, batch_size, steps)

        return predictions.sum(axis=1).sum(axis=1)
