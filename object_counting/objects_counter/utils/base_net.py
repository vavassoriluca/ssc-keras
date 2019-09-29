
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from object_classification.keras_models.resnet152 import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.models import Model


class BaseNet:

    """The :class:`BaseNet` object provides different implementations of CNN Classifiers used as Base Model to extract features from images.

    :param str model: the name of the classifier
    :param tuple input_shape: shape of the input
    """

    def __init__(self, model='vgg16', input_shape=(224,224,3)):

        self.input_shape = input_shape

        if model == 'vgg16':
            self.model = VGG16(input_shape=input_shape, weights='imagenet')
        elif model == 'mobilenet':
            self.model = MobileNetV2(input_shape=input_shape, weights='imagenet')
        elif model == 'resnet50':
            self.model = ResNet50(input_shape=input_shape, weights='imagenet')
        elif model == 'resnet152':
            self.model = ResNet152(input_shape=input_shape, weights='imagenet')
        elif model == 'inceptionresnet':
            self.model = InceptionResNetV2(input_shape=input_shape,  weights='imagenet')
        else:
            raise ValueError("The model you want to initialize as base is unknown.")

    def get_model(self):

        """This method return a Keras Model based on the Classifier model without the last classification layer.

        :returns: model without the last layer
        :rtype: keras.models.Model
        """

        return Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)
