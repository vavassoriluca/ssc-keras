
from object_counting.objects_counter.models.counting_model import CountingModel

from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import TimeDistributed
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers import LSTM

from object_counting.objects_counter.custom_layers.lstm_sequence import LSTMInputSequence

from keras import backend as K

import numpy as np

class SeqSub(CountingModel):

    """The :class:`SeqSub` model performs sequential subitizing count of an image. The output is for the full image but split by cell,
    so it must be summed up among all the cell of an image at the end of the prediction.

    :param tuple input_shape: shape of the input
    :param int num_classes: num of output classes
    :param num_bilstms: number of bi-directional LSTMs to capture context
    :param hidden_size: number of activations in the hidden layers
    :param include_relu: whether to include relu after the final fully-connected layer
    """

    def __init__(self, input_shape=(224,224,3), num_classes=20, num_bilstms=1, hidden_size=500, include_relu=False, weights=None):

        super().__init__(input_shape, num_classes=num_classes)

        input1 = Input(self.input_shape)
        c = []
        for i in range(2):
            #bilstm = Lambda(lambda x: x[:, i, ...])(input1)
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
        self.model = Model(inputs=input1, outputs=model1)

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
