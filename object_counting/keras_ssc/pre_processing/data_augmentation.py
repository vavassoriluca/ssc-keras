import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

from object_counting.keras_ssc.bounding_box_utils.bboxes_parser import parse_bboxes, encode_bboxes
from itertools import permutations

# https://github.com/Paperspace/DataAugmentationForObjectDetection

class DataAugmenter(object):


    def __init__(self, imgs_dir, annotations_dir, classes=None):

        """Instantiate Augmenter

        Parameters
        ----------
        img_dir: string path
            The directory containing the images

        annotations_dir: string path
            The directory containing the annotations

        img_dir: list of unique ordered strings
            The list containing the labels for the custom classes, ordered by id and unique
            Default list if it is None

        """

        self.imgs_dir = os.path.join(imgs_dir)
        self.annotations_dir = os.path.join(annotations_dir)
        self.classes = ['screw_0', 'screw_1', 'screw_2', 'screw_3',
                        'screw_4', 'screw_5', 'screw_6', 'screw_7',
                        'screw_8', 'screw_9'] if classes is None else classes


    def __call__(self, transformation_dict={"h_flip": .3,
                                            "v_flip": .3,
                                            "scale": .3,
                                            "translate": .2,
                                            "brightness": .2,
                                            "noise": (.05, .2),
                                            "swap_ch": .2
                                            }, copies_per_image=1):

        """Parse the annotations xml file

        Parameters
        ----------
        transforamtion_dict: dict
            The dictionary containing the possible tranformations:

                    h_flip:     horizontal flip, accepts float between (0,1) or tuple with range between (-1, 1)
                    v_flip:     vertical flip, accepts float between (0,1) or tuple with range between (-1, 1)
                    scale:      scale the image, accepts float > 0 or tuple with interval
                    translate:  translate the image, accepts float between (0,1) or tuple with range between (-1, 1)
                    brightness: change image's brightness, accepts float between (0,1) or tuple with range between (-1, 1)
                    noise:      add random noise to the image, accepts a tuple with density of the noise and probability
                                to apply it to an image
                    swap_ch:    swap the channels of the image, accepts probabilty of swapping them

                    When the method accepts a tuple with the rane, if a single float x is provided, the range will be (-x, x)

        """

        for image_name in os.listdir(os.path.join(self.imgs_dir)):
            image = cv2.imread(self.imgs_dir + image_name)[:, :, ::-1]
            # self.plot_image(image, bboxes)

            for i in range(copies_per_image):
                t_image = image.copy()
                annotations = ET.parse(self.annotations_dir + image_name.replace(".jpg", ".xml"))
                t_bboxes = parse_bboxes(annotations)

                for t, p in transformation_dict.items():
                    t_image, t_bboxes = getattr(self, t)(p, t_image, t_bboxes)

                #self.plot_image(t_image, t_bboxes)
                cv2.imwrite(self.imgs_dir + image_name.replace(".jpg", "aug{}.jpg".format(i)), t_image)
                new_xml = encode_bboxes(annotations, t_bboxes, image_name.replace(".jpg", "aug{}.jpg".format(i)))
                new_xml.write(self.annotations_dir + image_name.replace(".jpg", "aug{}.xml".format(i)))


    def h_flip(self, p, img, bboxes):

        """Randomly horizontally flips the Image with the probability *p*

        Parameters
        ----------
        p: float
            The probability with which the image is flipped


        Returns
        -------

        numpy.ndaaray
            Flipped image in the numpy format of shape `HxWxC`

        numpy.ndarray
            Tranformed bounding box co-ordinates of the format `n x 4` where n is
            number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

        """

        if random.random() < p:
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        return img, bboxes


    def v_flip(self, p, img, bboxes):

        """Randomly vertically flips the Image with the probability *p*

        Parameters
        ----------
        p: float
            The probability with which the image is flipped


        Returns
        -------

        numpy.ndaaray
            Flipped image in the numpy format of shape `HxWxC`

        numpy.ndarray
            Tranformed bounding box co-ordinates of the format `n x 4` where n is
            number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

        """

        if random.random() < p:
            img_center = np.array(img.shape[:2])[::-1] / 2
            img_center = np.hstack((img_center, img_center))
            img = img[::-1, :, :]
            bboxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - bboxes[:, [1, 3]])

            box_h = abs(bboxes[:, 1] - bboxes[:, 3])

            bboxes[:, 1] -= box_h
            bboxes[:, 3] += box_h

        return img, bboxes


    def translate(self, p, img, bboxes):

        """Randomly Translates the image

        Bounding boxes which have an area of less than 20% remaining in the
        transformed image is dropped. The resolution is maintained, and the remaining
        area if any is filled by black color.

        Parameters
        ----------
        translate: float or tuple(float)
            if **float**, the image is translated by a factor drawn
            randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
            `translate` is drawn randomly from values specified by the
            tuple

        Returns
        -------

        numpy.ndaaray
            Translated image in the numpy format of shape `HxWxC`

        numpy.ndarray
            Tranformed bounding box co-ordinates of the format `n x 5` where n is
            number of bounding boxes and 5 represents `x1,y1,x2,y2,c` of the box

        """

        img_shape = img.shape

        if type(p) == tuple:
            assert len(p) == 2, "Invalid range"
            assert p[0] > 0 and p[0] < 1
            assert p[1] > 0 and p[1] < 1
        else:
            assert p > 0 and p < 1
            p = (-p, p)

        translate_factor_x = random.uniform(*p)
        translate_factor_y = random.uniform(*p)

        canvas = np.zeros(img_shape)

        # get the top-left corner co-ordinates of the shifted image
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])
        M = np.float32([[1, 0, corner_x], [0, 1, corner_y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        bboxes = self.clip_boxes(bboxes, np.array([0, 0, img_shape[1], img_shape[0]]), 0.25)

        return img, bboxes


    def scale(self, p, img, bboxes):

        """Randomly scales an image

        Bounding boxes which have an area of less than 25% in the remaining in the
        transformed image is dropped. The resolution is maintained, and the remaining
        area if any is filled by black color.

        Parameters
        ----------
        p: float or tuple(float)
            if **float**, the image is scaled by a factor drawn
            randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
            the `scale` is drawn randomly from values specified by the
            tuple

        Returns
        -------

        numpy.ndaaray
            Scaled image in the numpy format of shape `HxWxC`

        numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        """

        img_shape = img.shape

        if type(p) == tuple:
            assert len(p) == 2, "Invalid range"
            assert p[0] > -1, "Scale factor can't be less than -1"
            assert p[1] > -1, "Scale factor can't be less than -1"
        else:
            assert p > 0, "Please input a positive float"
            p = (max(-1, -p), p)

        scale_x = random.uniform(*p)
        scale_y = scale_x
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)
        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)
        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])
        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]
        img = canvas
        bboxes = self.clip_boxes(bboxes, np.array([0, 0, img_shape[1], img_shape[0]]), 0.20)

        return img, bboxes


    def clip_boxes(self, bboxes, clip_box, alpha):

        """Clip the bounding boxes to the borders of an image

        Parameters
        ----------

        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        clip_box: numpy.ndarray
            An array of shape (4,) specifying the diagonal co-ordinates of the image
            The coordinates are represented in the format `x1 y1 x2 y2`

        alpha: float
            If the fraction of a bounding box left in the image after being clipped is
            less than `alpha` the bounding box is dropped.

        Returns
        -------

        numpy.ndarray
            Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
            number of bounding boxes left are being clipped and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        """
        ar_ = np.copy(self.area(bboxes))
        x_min = np.maximum(bboxes[:, 0].copy(), clip_box[0]).reshape(-1, 1)
        y_min = np.maximum(bboxes[:, 1].copy(), clip_box[1]).reshape(-1, 1)
        x_max = np.minimum(bboxes[:, 2].copy(), clip_box[2]).reshape(-1, 1)
        y_max = np.minimum(bboxes[:, 3].copy(), clip_box[3]).reshape(-1, 1)

        bboxes = np.hstack((x_min, y_min, x_max, y_max, bboxes[:, 4:])).copy()
        delta_area = np.absolute(ar_ - self.area(bboxes)) / np.absolute(ar_)
        mask = (delta_area < 1 - alpha).astype(int)
        bboxes = bboxes[mask == 1, :]

        return bboxes


    def area(self, b):
        return np.multiply(np.absolute(b[:, 0] - b[:, 2]),
                           np.absolute(b[:, 1] - b[:, 3])).copy()

    def brightness(self, p, img, bboxes):

        if type(p) == tuple:
            assert len(p) == 2, "Invalid range"
            assert p[0] > 0 and p[0] < 1
            assert p[1] > 0 and p[1] < 1
        else:
            assert p > 0 and p < 1
            p = (-p, p)

        value = np.uint8(random.uniform(*p) * 255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
        hsv[:, :, 2] += value
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img, bboxes


    def noise(self, p, img, bboxes):

        assert len(p) == 2, "Invalid range"
        assert p[0] > 0, "Noise density can't be less than 0"
        assert p[1] > 0, "Noise probability can't be less than 0"

        if random.random() < p[1]:
            noise = np.zeros(img.shape, np.int8)
            cv2.randn(noise, np.zeros(3), np.ones(3) * 255 * p[0])
            mask = np.random.choice([0, 1], size=img.shape, p=[1 - p[0], p[0]])
            img = cv2.add(img, np.multiply(noise, mask), dtype=cv2.CV_8UC3)

        return img, bboxes


    def swap_ch(self, p, img, bboxes):

        if random.random() < p:
            idx = random.choice([perm for perm in permutations(range(3), 3)])
            img = img[:, :, idx]
        return img, bboxes


    def plot_image(self, img, bboxes):

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes))).tolist()

        plt.figure(figsize=(20, 12))
        plt.imshow(img)

        current_axis = plt.gca()

        for box in bboxes:
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            color = colors[int(box[4])]
            label = '{}'.format(self.classes[int(box[4])])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

        plt.show()

