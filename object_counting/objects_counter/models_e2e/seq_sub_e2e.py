from object_counting.objects_counter.custom_layers.conv_to_lstm import Conv2LSTM
from object_counting.objects_counter.custom_layers.lstm_sequence import LSTMInputSequence
from object_counting.objects_counter.models_e2e.counting_model_e2e import CountingModelE2E
from object_counting.objects_counter.utils.base_net_e2e import BaseNetE2E

from keras.models import Model
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import TimeDistributed
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import LSTM

import numpy as np


class SeqSubE2E(CountingModelE2E):

    """The :class:`SeqSub` model performs sequential subitizing count of an image. The output is for the full image but split by cell,
    so it must be summed up among all the cell of an image at the end of the prediction.

    :param tuple input_shape: shape of the input
    :param str base_model: the name of the network used to extract features
    :param int num_classes: num of output classes
    :param num_bilstms: number of bi-directional LSTMs to capture context
    :param hidden_size: number of activations in the hidden layers
    :param include_relu: whether to include relu after the final fully-connected layer
    :param boolean freeze_base_net: whether to set non-trainable or trainable the layers of the base_net
    """

    def __init__(self, input_shape, base_model='resnet152', num_classes=20, num_bilstms=1, hidden_size=500, include_relu=False, freeze_base_net=True, grid7=False, weights=None):

        super().__init__(input_shape, base_model, num_classes=num_classes)

        self.base_model, added_layers = BaseNetE2E(base_model, input_shape, conv_output=True, conv7_output=grid7).get_model()

        freeze_len = len(self.base_model.layers) - added_layers

        input1 = Dense(hidden_size)(self.base_model.layers[-1].output)
        input1 = Conv2LSTM()(input1)
        c = []
        for i in range(2):
            bilstm = LSTMInputSequence(i)(input1)
            bilstm = TimeDistributed(ReLU())(bilstm)
            bilstm = TimeDistributed(Dense(hidden_size))(bilstm)
            bilstm = TimeDistributed(ReLU())(bilstm)
            for j in range(num_bilstms):
                lstm_size = hidden_size * np.power(2, j)
                bilstm = Bidirectional(LSTM(lstm_size, return_sequences=True))(bilstm)
            c.append(bilstm)
        model1 = Concatenate()(c)
        model1 = ReLU()(model1)
        model1 = Dense(self.num_classes)(model1)
        if include_relu != 0:
            model1 = TimeDistributed(ReLU())(model1)
        self.model = Model(inputs=self.base_model.input, outputs=model1)

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

        return predictions.sum(axis=1)
