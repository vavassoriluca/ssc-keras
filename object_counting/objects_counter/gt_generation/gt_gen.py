from __future__ import division

import numpy as np
from scipy.stats import mvn
from object_counting.objects_counter.gt_generation.gt_utils import GTUtils
from utils.annotation_parser import AnnotationParser
from object_counting.objects_counter.gt_generation.cython.bboxes_to_count import bboxes2gt


class GTGen:

    """
    The :class:`GTGen` class provide a method to convert bounding boxes notation stored
    in different formats into a count notation.
    It is possible to define a grid division to split the count of an object between the cells.
    (See also Subitizing models).
    """

    @staticmethod
    def get_gt(classes, filepath, annotations_ext='xml', input_shape=None, grid_division=None, mode='linear',
               std_ratio=(6, 6)):

        """
        This method convert bounding boxes in the count per class per grid.

        :param list classes: list of strings, the names of the classes
        :param str filepath: path to the annotation file
        :param str annotations_ext: extension of the file
        :param tuple input_shape: shape of the image
        :param tuple grid_division: division in cells of the image (x_axis, y_axis)
        :param str mode: the approach used to split the count, either linear, normal (gaussian) or active_contour
        :param tuple std_ratio: tuple to compute std along the 2 axes, it is calculated as side_length/std_ratio

        :return: array with the count per class [per cell]
        :rtype: np.ndarray
        """

        assert annotations_ext in ['xml', 'txt', 'json'], "Annotation extension not valid"

        bboxes = AnnotationParser().parse(filepath, classes, file_ext='xml')

        if input_shape is None or grid_division is None:

            counter = np.zeros(len(classes))
            for bbox in bboxes:
                counter[int(bbox[0])] += 1

        else:

            if mode == 'linear':
                counter = bboxes2gt(np.array(input_shape), bboxes, len(classes), np.array(grid_division))
            elif mode == 'normal':
                counter = GTGen.bboxes2gt_normal(classes, input_shape, bboxes, grid_division, std_ratio)
            else:
                raise ValueError('Unknown GT generation mode.')

        return counter

    @staticmethod
    def bboxes2gt_normal(classes, input_shape, bboxes, grid_division, std_ratio=(6, 6)):

        """This method

        :param list classes: list of strings, the names of the classes
        :param tuple input_shape: shape of the image
        :param np.ndarray bboxes: list of bboxes in the format class_id, xmin, ymin, xmax, ymax
        :param grid_division: division in cells of the image (x_axis, y_axis)
        :param std_ratio: tuple to compute std along the 2 axes, it is calculated as side_length/std_ratio

        :return: array with the count per class per cell
        :rtype: np.ndarray
        """

        grids = GTGen.get_grids(input_shape, grid_division)
        counter = np.zeros((grids.shape[0], len(classes)))

        for b in range(len(bboxes)):
            bbox = bboxes[b]
            # Define mu and covariance matrix
            mu = np.array([(bbox[3] - bbox[1]) / 2 + bbox[1], (bbox[4] - bbox[2]) / 2 + bbox[2]])
            S = np.array([[((bbox[3] - bbox[1]) / std_ratio[0]) ** 2, 0],
                          [0, ((bbox[4] - bbox[2]) / std_ratio[1]) ** 2]])
            norm_coeff = mvn.mvnun(np.array(bbox[1:3]), np.array(bbox[3:]), mu, S)[0]

            if bbox[1] > input_shape[1] or bbox[3] > input_shape[1] or bbox[2] > input_shape[0] or bbox[4] > \
                    input_shape[0]:
                raise ValueError("Bounding Boxes coordinates exceeds the image boundaries")
            if any(bbox) < 0:
                raise ValueError("Found negative Bounding Boxes coordinates")
            for i in range(grids.shape[0]):
                low, upp = GTUtils.coordinates_intersection(grids[i], bbox[1:])
                if low is None:
                    continue
                counter[i, int(bbox[0])] += mvn.mvnun(low, upp, mu, S)[0] / norm_coeff

        return counter

    @staticmethod
    def get_grids(input_shape, grid_division):

        """This method return a grid division of an image providing the coordinates of each cell.

        :param tuple input_shape: shape of the original image
        :param tuple grid_division: tuple with the number of divisions per dimension

        :return: an array containing the coordinates of the grids
        :rtype: np.ndarray
        """

        x_coords = np.linspace(0, input_shape[1], grid_division[0] + 1)
        y_coords = np.linspace(0, input_shape[0], grid_division[1] + 1)
        grids = []
        for i in range(grid_division[0]):
            for j in range(grid_division[1]):
                grids.append([x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]])
        return np.array(grids)



