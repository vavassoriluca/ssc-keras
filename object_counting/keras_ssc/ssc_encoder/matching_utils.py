"""
Utilities to match ground truth boxes to anchor boxes.

Adapted code based on the version by Pierluigi Ferrari
"""

from __future__ import division
import numpy as np


def match_bipartite_greedy(weight_matrix):
    """
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    :param np.ndarray weight_matrix: A 2D Numpy array that represents the weight matrix
        for the matching process. If `(m,n)` is the shape of the weight matrix,
        it must be `m <= n`. The weights can be integers or floating point
        numbers. The matching process will maximize, i.e. larger weights are
        preferred over smaller weights.

    :return np.ndarry: A 1D Numpy array of length `weight_matrix.shape[0]` that represents
    the matched index along the second axis of `weight_matrix` for each index
    along the first axis.
    """

    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))  # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):

        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1)  # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)  # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches


def match_multi(weight_matrix, threshold):
    """
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective element should be set to a value below `threshold`.

    :param np.ndarray weight_matrix: A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
    :param float threshold: A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    :returns np.ndarray: Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    """

    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0)  # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]  # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met


def match_bipartite_greedy_distributed(weight_matrix, predictor_sizes, n_boxes, mode='linear', treshold=0.2):
    """
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    :param np.ndarray weight_matrix: A 2D Numpy array that represents the weight matrix
        for the matching process. If `(m,n)` is the shape of the weight matrix,
        it must be `m <= n`. The weights can be integers or floating point
        numbers. The matching process will maximize, i.e. larger weights are
        preferred over smaller weights.

    :return np.ndarray: A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    """

    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))  # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros((num_ground_truth_boxes, len(predictor_sizes)), dtype=np.int)
    overlaps_gt = np.zeros((num_ground_truth_boxes, len(predictor_sizes)))

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):

        pos = 0
        for i, pred_size in enumerate(predictor_sizes):
            if isinstance(n_boxes, list):
                n_box = pred_size[0]*pred_size[1]*n_boxes[i]
            else:
                n_box = pred_size[0]*pred_size[1]*n_boxes

            # Find the maximal anchor-ground truth pair in two steps: First, reduce
            # over the anchor boxes and then reduce over the ground truth boxes.
            anchor_indices = np.argmax(weight_matrix[:, pos:(pos + n_box)], axis=1) + pos  # Reduce along the anchor box axis.
            overlaps = weight_matrix[all_gt_indices, anchor_indices]
            ground_truth_index = np.argmax(overlaps)  # Reduce along the ground truth box axis.
            anchor_index = anchor_indices[ground_truth_index]
            overlaps_gt[ground_truth_index, i] = weight_matrix[ground_truth_index, anchor_index]  # Set the match.
            matches[ground_truth_index, i] = anchor_index  # Set the match.

            # Set the row of the matched ground truth box and the column of the matched
            # anchor box to all zeros. This ensures that those boxes will not be matched again,
            # because they will never be the best matches for any other boxes.
            weight_matrix[ground_truth_index] = 0
            weight_matrix[:, anchor_index] = 0
            pos = pos + n_box

    if np.any(overlaps_gt > treshold):
        overlaps_gt[overlaps_gt < treshold] = 0
    if mode is 'linear':
        overlaps_gt = np.exp(overlaps_gt) / np.sum(np.exp(overlaps_gt), axis=1)[:, np.newaxis]
    elif mode is 'softmax':
        row_sums = overlaps_gt.sum(axis=1)
        row_sums[row_sums == 0] = 1
        overlaps_gt = overlaps_gt / row_sums[:, np.newaxis]
    elif mode is 'pushmax':
        overlaps_gt = overlaps_gt / np.reciprocal(np.power(overlaps_gt, 3).clip(0.001))
        row_sums = overlaps_gt.sum(axis=1)
        row_sums[row_sums == 0] = 1
        overlaps_gt = overlaps_gt / row_sums[:, np.newaxis]
    else:
        raise ValueError('Unknown mode.')

    return matches, overlaps_gt
