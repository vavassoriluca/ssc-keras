from __future__ import division
import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from tensorflow import gather

class LSTMInputSequence(Layer):

    """The :class:`LSTMInputSequence` is a custom layer that takes a 3-dimensional tensor (batch_size, sequence_size, features) and
    return a tensor of the same shape and type as the input tensor, with a sorted sequence according a defined direction.

    :param int direction: either 0 or 1, 0 for ᴎ, 1 for Ƨ
    """

    def __init__(self, direction, **kwargs):

        self.direction = direction
        super(LSTMInputSequence, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(LSTMInputSequence, self).build(input_shape)

    def call(self, x, mask=None):

        """Overrides the :class:`keras.engine.topology.Layers` method. It orders the sequence according to the direction. The :metohd:`gather` method from Tensorflow has been used bypassing the Keras backend.

        :param x: input tensor
        :param mask: tensor mask
        :return: re-ordered tensor
        """

        cells = x.shape[1].value
        side = int(np.sqrt(cells))

        indices = []
        flag = True

        if self.direction == 0:
            for i in range(side):
                if flag:
                    indices.extend(range(i, cells, side))
                    flag = False
                else:
                    indices.extend(range(i, cells, side)[::-1])
                    flag = True
        elif self.direction == 1:
            for i in range(side):
                if flag:
                    indices.extend(range(i * side, (i + 1) * side))
                    flag = False
                else:
                    indices.extend(range(i * side, (i + 1) * side)[::-1])
                    flag = True

        indices = K.constant(indices, dtype=np.int32)
        output = gather(x, indices, axis=1)

        return output

    def get_config(self):
        config = {'direction': self.direction}
        base_config = super(LSTMInputSequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
