from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from object_classification.keras_models.resnet152 import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import ReLU
from keras.models import Model


class BaseNetE2E:

    def __init__(self, model='resnet152', input_shape=(224,224,3), conv_output=True, conv7_output=False):

        """The :class:`BaseNetE2E` object provides different implementations of CNN Classifiers used as Base Model to extract features from images.
        If :paramref:`utils.base_net_e2e.BaseNetE2E.conv_output is set to True, the Classifier is slightly changed to have a 3D feature map as output.

        :param str model: the name of the classifier
        :param tuple input_shape: shape of the input
        :param bool conv_output: whether to change the model to output a 3D feature map or a 1D flatten output
        :param bool freeze_layers: whether to set the layers from the Classifier as not trainable or trainable
        """

        self.input_shape = input_shape
        self.conv_output = conv_output
        self.added_layers = 9
        weights = 'imagenet'

        if conv7_output:
            self.added_layers = 0
            if model == 'vgg16':
                self.model = VGG16(input_shape=input_shape, weights=weights)
                self.conv_model = self.model.layers[-5].output
            elif model == 'mobilenet':
                self.model = MobileNetV2(input_shape=input_shape, weights=weights)
                self.conv_model = self.model.layers[-3].output
            elif model == 'resnet50':
                self.model = ResNet50(input_shape=input_shape, weights=weights)
                self.conv_model = self.model.layers[-3].output
            elif model == 'resnet152':
                self.model = ResNet152(input_shape=input_shape, weights=weights)
                self.conv_model = self.model.layers[-4].output
            elif model == 'inceptionresnet':
                self.model = InceptionResNetV2(input_shape=input_shape, weights=weights)
                self.conv_model = Conv2D(1024, (2, 2), padding='valid', kernel_initializer='he_normal',
                                         name='output1')(self.model.layers[-4].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.added_layers = 3
        elif conv_output:
            if model == 'vgg16':
                self.model = VGG16(input_shape=input_shape, weights=weights)
                self.conv_model = Conv2D(1024, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='fc6')(self.model.layers[-5].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(1024, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='fc7')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                         name='bottleneck_output')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
            elif model == 'mobilenet':
                self.model = MobileNetV2(input_shape=input_shape, weights=weights)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output1')(self.model.layers[-3].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output2')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                         name='bottleneck_output')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
            elif model == 'resnet50':
                self.model = ResNet50(input_shape=input_shape, weights=weights)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output1')(self.model.layers[-3].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output2')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                         name='bottleneck_output')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
            elif model == 'resnet152':
                self.model = ResNet152(input_shape=input_shape, weights=weights)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output1')(self.model.layers[-4].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output2')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                         name='bottleneck_output')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
            elif model == 'inceptionresnet':
                self.model = InceptionResNetV2(input_shape=input_shape,  weights=weights)
                self.conv_model = Conv2D(1024, (3, 3), padding='valid', kernel_initializer='he_normal',
                                         name='output1')(self.model.layers[-4].output)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.conv_model = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                         name='bottleneck_output')(self.conv_model)
                self.conv_model = BatchNormalization()(self.conv_model)
                self.conv_model = ReLU()(self.conv_model)
                self.added_layers = 6
            else:
                raise ValueError("The model you want to initialize as base is unknown.")
        else:
            if model == 'vgg16':
                self.model = VGG16(input_shape=input_shape, weights=weights)
            elif model == 'mobilenet':
                self.model = MobileNetV2(input_shape=input_shape, weights=weights)
            elif model == 'resnet50':
                self.model = ResNet50(input_shape=input_shape, weights=weights)
            elif model == 'resnet152':
                self.model = ResNet152(input_shape=input_shape, weights=weights)
            elif model == 'inceptionresnet':
                self.model = InceptionResNetV2(input_shape=input_shape,  weights=weights)
            else:
                raise ValueError("The model you want to initialize as base is unknown.")

    def get_model(self):

        """This method return a Keras Model based on the Classifier model.

        :returns: model without the last layer
        :rtype: keras.models.Model
        """

        if self.conv_output:
            return Model(inputs=self.model.inputs, outputs=self.conv_model), self.added_layers
        else:
            return Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

BaseNetE2E(conv_output=False).get_model().summary()