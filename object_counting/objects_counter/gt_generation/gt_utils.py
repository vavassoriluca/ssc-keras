import numpy as np


class GTUtils:

    """
    The :class:`GTUtils` provides stati methods to support Groud Truth generation.
    """

    @staticmethod
    def area(a):

        """This method return the area of a rectangle given the coordinates of the top left and bottom right corners.

        :param list a: list of
        :return: area of a rectangular region
        :rtype: float
        """

        return (a[2] - a[0]) * (a[3] - a[1])

    @staticmethod
    def intersection_area(a, b):

        """This method return the area of the intersection of two rectangles.

        :param list a: coordinates of rectangle A encoded in the format [xmin, ymin, xmax, ymax]
        :param list b: coordinates of rectangle B encoded in the format [xmin, ymin, xmax, ymax]

        :return: intersection area
        :rtype: float
        """
        xmin = max(a[0], b[0])
        ymin = max(a[1], b[1])
        xmax = min(a[2], b[2])
        ymax = min(a[3], b[3])

        if xmin < xmax and ymin < ymax:
            return (xmax - xmin) * (ymax - ymin)
        else:
            return .0

    @staticmethod
    def coordinates_intersection(a, b):

        """This method return the area of the intersection of two rectangles.

        :param list a: coordinates of rectangle A encoded in the format [xmin, ymin, xmax, ymax]
        :param list b: coordinates of rectangle B encoded in the format [xmin, ymin, xmax, ymax]

        :return: intersection area
        :rtype: lists of coordinates
        """

        if a[0] > b[2] or a[1] > b[3] or a[2] < b[0] or a[3] < b[1]:
            return None, None
        else:
            return np.array([max(a[0], b[0]), max(a[1], b[1])]), np.array([min(a[2], b[2]), min(a[3], b[3])])