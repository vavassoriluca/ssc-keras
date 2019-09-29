
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from object_classification.keras_models.resnet152 import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.models import Model


class BaseNet:

    """The :class:`BaseNet` object provides different implementations of CNN Classifiers either used to classify or as the Base Model to extract features from images.

    :param str model: the name of the classifier
    :param tuple input_shape: shape of the input
    :param str weights: path to the weights of the model, the name of the layers must be the same
    :param bool include_top: whether to include the classification layer or not
    :param bool include_top: whether to return the output of the last convolutional layer or not
    """

    def __init__(self, model='vgg16', input_shape=(224, 224, 3), weights='imagenet', include_top=True, conv_output=False):

        assert (conv_output and not include_top) or (not conv_output and include_top)\
               or (not conv_output and not include_top), 'Either include_top or conv_output could be True, not both.'

        self.input_shape = input_shape
        self.include_top = include_top
        self.conv_output = conv_output
        self.last_conv_layer = -1

        if model == 'vgg16':
            self.model = VGG16(input_shape=input_shape, weights=weights, include_top=include_top)
            self.last_conv_layer = -2
        elif model == 'mobilenet':
            self.model = MobileNetV2(input_shape=input_shape, weights=weights, include_top=include_top)
        elif model == 'resnet50':
            self.model = ResNet50(input_shape=input_shape, weights=weights, include_top=include_top)
        elif model == 'resnet152':
            self.model = ResNet152(input_shape=input_shape, weights=weights, include_top=include_top)
        elif model == 'inceptionresnet':
            self.model = InceptionResNetV2(input_shape=input_shape, weights=weights, include_top=include_top)
        else:
            raise ValueError("The model you want to initialize as base is unknown.")

        if weights is not None and weights is not 'imagenet':
            self.model.load_weights(weights, by_name=True)

    def get_model(self):

        """This method return a Keras Model based on the Classifier model without the last classification layer.

        :returns: model without the last layer
        :rtype: keras.models.Model
        """

        if self.conv_output:
            return Model(inputs=self.model.inputs, outputs=self.model.layers[self.last_conv_layer].output)
        else:
            return Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)



