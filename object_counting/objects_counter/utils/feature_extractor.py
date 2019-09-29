import cv2
from keras.models import Model
from object_counting.objects_counter.utils.base_net import BaseNet

import numpy as np

# from object_counting.objects_counter.utils.cython.feature_extractor_cython import generate_features_grid_cython


class FeatureExtractor:

    """The :class:`FeatureExtractor` object, given a set of images, provides methods to extract their features through one of the proposed CNN Classifier instantiated as base model.
    If a grid division is given, it extracts features per cell.

    :param str feature_model: name of the CNN Classifier to extract features
    :param tuple input_shape: shape of the image accepted by the base model
    """

    def __init__(self, feature_model, input_shape=(224, 224, 3), base_weights=None):

        self.input_shape = input_shape
        self.base_model = BaseNet(model=feature_model, input_shape=self.input_shape).get_model()
        if base_weights is not None:
            try:
                self.base_model.load_weights(base_weights, by_name=True)
            except:
                print("Uncompatible weights")

    def extract(self, image, grid_division=(3, 3)):

        """This method extracts the feature for the image, either for the entire image or for each cell according to the grid division

        :param np.ndarray image: image to be processed
        :param tuple grid_division: division of the image in cells. Set to None if no division is needed
        :return: features per image/cell
        :rtype: np.ndarray
        """

        if grid_division is not None:
            # return generate_features_grid_cython(image, grid_division, self.input_shape, self.base_net)
            return self.generate_features_grid(image, grid_division)
        else:
            return self.generate_features_full(image)

    def generate_features_grid(self, image, grid_division):

        """This method extract the features for the image, dividing it in cells according to the grid division

        :param np.ndarray image: image to be processed
        :param tuple grid_division: tuple, division of the image in cells. Set to None if no division is needed
        :return: features per cell
        :rtype: np.ndarray

        """

        features = []
        x_coords = np.linspace(0, image.shape[1], grid_division[0] + 1)
        y_coords = np.linspace(0, image.shape[0], grid_division[1] + 1)
        for i in range(grid_division[0]):
            for j in range(grid_division[1]):
                crop = image[int(y_coords[j]):int(y_coords[j+1]), int(x_coords[i]):int(x_coords[i+1])].copy()
                try:
                    crop = cv2.resize(crop, self.input_shape[:2])
                except Exception as e:
                    print(e, crop.shape, image.shape, y_coords, x_coords)
                feature = self.base_model.predict(np.expand_dims(crop, axis=0))
                features.append(feature.reshape(feature.size))
        return np.array(features)

    def generate_features_full(self, image):

        """This method extracts the features for the image as is, it resizes the image according to the required input shape.

        :param np.ndarray image: image to be processed
        :return: the features of the image
        :rtype: np.ndarray
        """

        image = cv2.resize(image, self.input_shape[:2])
        features = self.base_model.predict(np.expand_dims(image, axis=0))
        return features.reshape(features.size)
