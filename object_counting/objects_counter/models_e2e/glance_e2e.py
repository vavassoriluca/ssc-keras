from object_counting.objects_counter.utils.base_net_e2e import BaseNetE2E
from object_counting.objects_counter.models_e2e.counting_model_e2e import CountingModelE2E

from keras.models import Model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import ReLU


class GlanceE2E(CountingModelE2E):

    def __init__(self, input_shape=(224, 224, 3), base_model='resnet152', num_classes=20, num_hidden=1, hidden_size=500, include_relu=False, freeze_base_net=True, weights=None):

        """
        Initialize input-output dimensionalities

        :param tuple input_shape: shape of the input
        :param str base_model: the name of the network used to extract features
        :param int num_classes: num of output classes
        :param num_hidden: number of hidden layers in the MLP
        :param hidden_size: number of activations in the hidden layers in the MLP
        :param include_relu: whether to include relu after the final fully-connected layer
        :param boolean freeze_base_net: whether to set non-trainable or trainable the layers of the base_net
        """

        super().__init__(input_shape, base_model, num_classes)

        self.base_model = BaseNetE2E(base_model, input_shape, conv_output=False).get_model()

        freeze_len = len(self.base_model.layers)

        model_input = self.base_model.layers[-1].output
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

        self.model = Model(inputs=self.base_model.input, outputs=model)

        if freeze_base_net:
            for l in self.model.layers[:freeze_len]:
                l.trainable = False

