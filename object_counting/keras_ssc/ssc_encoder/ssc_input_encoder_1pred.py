"""
An encoder that converts ground truth annotations to SSC-compatible training targets.

Adapted code based on the version by Pierluigi Ferrari
"""

from __future__ import division
import numpy as np


class SSCInputEncoder1Pred:

    """
    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSC model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.

    :param  int n_classes: The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
    :param bool diagnostics: If `True`, not only the encoded ground truth tensor will be returned,
        but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
        This can be very useful if you want to visualize which anchor boxes got matched to which ground truth boxes.
    """

    def __init__(self,
                 n_classes,
                 diagnostics=False):

        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.n_classes = n_classes  # + 1 for the background class
        self.diagnostics = diagnostics

    def __call__(self, ground_truth_labels):

        """
        Converts ground truth bounding box data into a suitable format to train an SSC model.

        :param list ground_truth_labels: A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.

        :return y_encoded: a 2D numpy array of shape `(batch_size, #classes)` that serves as the
            ground truth label tensor for training.
        """

        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        ##################################################################################
        # Compute the count out of the bounding boxes
        ##################################################################################

        y_encoded = np.zeros((batch_size, self.n_classes))

        class_vectors = np.eye(self.n_classes)  # An identity matrix that we'll use as one-hot class vectors

        for i in range(batch_size):  # For each batch item...

            if ground_truth_labels[i].size == 0:
                continue  # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float)  # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]  # The one-hot class IDs for the ground truth boxes of this batch item

            # Write the ground truth data
            y_encoded[i] = classes_one_hot.sum(axis=0)

        if self.diagnostics:
            return y_encoded, ground_truth_labels
        return y_encoded


class DegenerateBoxError(Exception):
    """
    An exception class to be raised if degenerate boxes are being detected.
    """
    pass
