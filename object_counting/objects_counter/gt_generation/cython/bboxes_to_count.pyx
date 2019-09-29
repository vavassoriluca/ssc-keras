import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef get_grids(long[:] input_shape, long[:] grid_division):

    """This method return a grid division of an image providing the coordinates of each cell.

    :param tuple input_shape: shape of the original image
    :param tuple grid_division: tuple with the number of divisions per dimension

    :return: an array containing the coordinates of the grids
    :rtype: np.ndarray
    """
    cdef double[:,:]grids = np.zeros((grid_division[0]*grid_division[1],4))
    cdef double[:] x_coords = np.linspace(0, input_shape[1], grid_division[0] + 1)
    cdef double[:] y_coords = np.linspace(0, input_shape[0], grid_division[1] + 1)

    cdef int i = 0
    cdef int j = 0
    cdef long idx = 0

    for i in range(grid_division[0]):
        for j in range(grid_division[1]):
            idx = i*grid_division[0] + j
            grids[idx][0] = x_coords[i]
            grids[idx][1] = y_coords[j]
            grids[idx][2] = x_coords[i + 1]
            grids[idx][3] = y_coords[j + 1]

    return grids

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef get_grids_e2e(long[:] input_shape, long[:] grid_division):

    """This method return a grid division of an image providing the coordinates of each cell.

    :param tuple input_shape: shape of the original image
    :param tuple grid_division: tuple with the number of divisions per dimension

    :return: an array containing the coordinates of the grids
    :rtype: np.ndarray
    """
    cdef double[:,:,:] grids = np.zeros((grid_division[0],grid_division[1],4))
    cdef double[:] x_coords = np.linspace(0, input_shape[1], grid_division[0] + 1)
    cdef double[:] y_coords = np.linspace(0, input_shape[0], grid_division[1] + 1)

    cdef int i = 0
    cdef int j = 0
    cdef long idx = 0

    for i in range(grid_division[0]):
        for j in range(grid_division[1]):
            grids[j,i,0] = x_coords[i]
            grids[j,i,1] = y_coords[j]
            grids[j,i,2] = x_coords[i + 1]
            grids[j,i,3] = y_coords[j + 1]

    return grids

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef area(double[:] A):

    """This method return the area of a rectangle given the coordinates of the top left and bottom right corners.

    :param list A: list of
    :return: area of a rectangular region
    :rtype: float
    """

    return (A[2] - A[0]) * (A[3] - A[1])

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef intersection_area(double[:] A, double[:] B):

    """This method return the area of the intersection of two rectangles.

    :param A: coordinates of rectangle A encoded in the format [xmin, ymin, xmax, ymax]
    :param B: coordinates of rectangle B encoded in the format [xmin, ymin, xmax, ymax]

    :return: intersection area
    :rtype: float
    """
    cdef float xmin = max(A[0], B[0])
    cdef float ymin = max(A[1], B[1])
    cdef float xmax = min(A[2], B[2])
    cdef float ymax = min(A[3], B[3])

    if xmin < xmax and ymin < ymax:
        return (xmax - xmin) * (ymax - ymin)
    else:
        return .0


def bboxes2gt(long[:] input_shape, double[:,:] bboxes, long classes, long[:] grid_division):


    cdef double[:,:] grids = get_grids(input_shape, grid_division)
    cdef double[:,:] counter = np.zeros((grids.shape[0], classes))

    cdef double[:] bbox
    cdef double[:] grid

    cdef int b = 0
    cdef int i = 0

    for b in range(bboxes.shape[0]):
        bbox = bboxes[b]
        if bbox[1] > input_shape[1] or bbox[3] > input_shape[1] or bbox[2] > input_shape[0] or bbox[4] > input_shape[0]:
            raise ValueError("Bounding Boxes coordinates exceeds the image boundaries")
        if any(bbox) < 0:
            raise ValueError("Found negative Bounding Boxes coordinates")
        for i in range(grids.shape[0]):
            counter[i, int(bbox[0])] += intersection_area(grids[i, :], bbox[1:]) / area(bbox[1:])

    return np.array(counter)

def bboxes2gt_e2e(long[:] input_shape, double[:,:] bboxes, long classes, long[:] grid_division):


    cdef double[:,:,:] grids = get_grids_e2e(input_shape, grid_division)
    cdef double[:,:,:] counter = np.zeros((grids.shape[0], grids.shape[1], classes))

    cdef double[:] bbox
    cdef double[:] grid

    cdef int b = 0
    cdef int i = 0
    cdef int j = 0

    for b in range(bboxes.shape[0]):
        bbox = bboxes[b]
        if bbox[1] > input_shape[1] or bbox[3] > input_shape[1] or bbox[2] > input_shape[0] or bbox[4] > input_shape[0]:
            raise ValueError("Bounding Boxes coordinates exceeds the image boundaries")
        if any(bbox) < 0:
            raise ValueError("Found negative Bounding Boxes coordinates")
        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):
                counter[i, j, int(bbox[0])] += intersection_area(grids[i, j, :], bbox[1:]) / area(bbox[1:])

    return np.array(counter)