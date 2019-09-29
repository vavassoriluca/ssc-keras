from bs4 import BeautifulSoup
import numpy as np


class AnnotationParser:

    """The :class:`AnnotationParser` object provides methods to parse different bounding boxes annotation files in the following formats:

    - xml - PASCAL VOC Dataset notation
    - txt - COCO Dataset notation
    - json - to be implemented
    """

    @staticmethod
    def parse(filepath, classes, file_ext='xml'):

        """Parse the file provided, if existing

        :param str filepath: path to file to parse
        :param classes: the names of the classes
        :type classes: list of strings
        :param str file_ext: extension of the files to parse

        :return: bounding boxes of the image
        :rtype: np.ndarray
        """

        assert file_ext in ['xml', 'txt', 'json']

        if file_ext == 'txt':
            return AnnotationParser.parse_txt(filepath, classes)
        elif file_ext == 'xml':
            return AnnotationParser.parse_xml(filepath, classes)
        elif file_ext == 'json':
            return AnnotationParser.parse_json(filepath, classes)

    @staticmethod
    def parse_xml(filepath, classes):

        """This method parse bounding boxes saved in PASCAL VOC format

        :param filepath: string, path to file to parse

        :return: bounding boxes of the image
        :rtype: np.ndarray
        """

        with open(filepath) as f:
            soup = BeautifulSoup(f, 'xml')

        boxes = []  # We'll store all boxes for this image here.
        objects = soup.find_all('object')  # Get a list of all objects in this image.

        # Parse the data for each object.
        for obj in objects:
            class_name = obj.find('name', recursive=False).text
            class_id = float(classes.index(class_name))
            # Get the bounding box coordinates.
            bndbox = obj.find('bndbox', recursive=False)
            xmin = float(bndbox.xmin.text)
            ymin = float(bndbox.ymin.text)
            xmax = float(bndbox.xmax.text)
            ymax = float(bndbox.ymax.text)

            boxes.append([class_id, xmin, ymin, xmax, ymax])

        return np.array(boxes)

    @staticmethod
    def parse_txt(filepath, classes):

        """This method parse bounding boxes saved in COCO format

        :param str filepath: path to file to parse

        :return: bounding boxes of the image
        :rtype: np.ndarray
        """

        boxes = []  # We'll store all boxes for this image here.
        with open(filepath,'r') as f:
            for line in f.read().splitlines():
                box = line.split()
                if str.isdigit(box[0]):
                    boxes.append(list(map(float, box)))
                else:
                    box[0] = classes.index(box[0])
                    boxes.append(list(map(float, box)))

        return np.array(boxes)

    @staticmethod
    def parse_json(filepath, classes):

        # TODO
        pass