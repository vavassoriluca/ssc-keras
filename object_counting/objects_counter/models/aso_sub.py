from __future__ import division
import numpy as np

from object_counting.objects_counter.models.counting_model import CountingModel

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import ReLU


class AsoSub(CountingModel):

    """The :class:`AsoSub` model performs associative subitizing count of an image. The output is for a single cell,
    so it must be summed up among all the cell of an image at the end of the prediction.

    The model consists of the same architecture of :class:`models.glance.Glance`, with the only difference that the input is an individual cell rather then the whole image.


    :param input_shape: tuple, shape of the input
    :param int num_hidden: number of hidden layers in the MLP
    :param int hidden_size: number of activations in the hidden layers in the MLP
    :param bool include_relu: whether to include relu after the final fully-connected layer
    :param num_classes: int, num of output classes
    """

    def __init__(self, input_shape, num_classes=20, num_hidden=1, hidden_size=500, include_relu=False, weights=None):

        super().__init__(input_shape, num_classes=num_classes)

        model_input = Input(shape=input_shape)
        model = Dense(hidden_size)(model_input)
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

        self.model = Model(inputs=model_input, outputs=model)

    def predict(self,
                X,
                batch_size=None,
                steps=None):

        """This method overrides the :meth:`CountingModelE2E.predict`. It perform a normal prediction and finally decode the output to compute the global count

        :param np.ndarray X: samples, first dimension number of samples, second dimension number of cells
        :param int batch_size: size of a batch
        :param int steps: number of batches to process
        """

        predictions = np.reshape(super().predict(X, batch_size, steps), (-1, 9, self.num_classes))

        return predictions.sum(axis=1)










