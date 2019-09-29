from __future__ import division
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

class Conv2LSTM(Layer):

    """The :class:`Conv2LSTM` is a custom layer that reshapes the input tensor collapsing the width and height dimensions to a single dimension that represents the sequence accepted by the LSTM.
    """

    def __init__(self, **kwargs):
        super(Conv2LSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(Conv2LSTM, self).build(input_shape)

    def call(self, x, mask=None):

        """Overrides the :class:`keras.engine.topology.Layers` method. It collapses the second and third dimension of the tensor into a single dimension.

        :param x: input tensor
        :param mask: tensor mask
        :return: re-ordered tensor
        """

        return K.reshape(x, (-1, x.shape[1].value * x.shape[2].value, x.shape[3].value))

    def get_config(self):
        base_config = super(Conv2LSTM, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (input_shape[1]*input_shape[2], input_shape[3])
