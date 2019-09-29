
from object_counting.objects_counter.models.counting_model import CountingModel

from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import ReLU

class Glance(CountingModel):

    """The :class:`Glance` model provides an object to perform glance count of an image. The output is for the whole image.

    :param tuple input_shape: input shape
    :param int num_classes: number of output classes
    :param num_hidden: int, number of hidden layers in the MLP
    :param hidden_size: int, number of activations in the hidden layers in the MLP
    :param include_relu: boolean, whether to include relu after the final fully-connected layer
    """

    def __init__(self, input_shape=(224,224,3), num_classes=20, num_hidden=1, hidden_size=500, include_relu=False, weights=None):

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
