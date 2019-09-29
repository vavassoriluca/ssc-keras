import numpy as np
import cv2
cimport numpy as np
cimport cython

def generate_features_grid_cython(image, grid_division, input_shape, base_net):

    cdef int i = 0
    cdef int j = 0

    cdef long[:] grid_div = np.array(grid_division)

    features = []
    cdef double[:] x_coords = np.linspace(0, image.shape[1], grid_division[0] + 1)
    cdef double[:] y_coords = np.linspace(0, image.shape[0], grid_division[1] + 1)

    for i in range(grid_div[0]):
        for j in range(grid_div[1]):
            crop = image[int(y_coords[j]):int(y_coords[j + 1]), int(x_coords[i]):int(x_coords[i + 1])].copy()
            try:
                crop = cv2.resize(crop, input_shape[:2])
            except Exception as e:
                print(e, crop.shape, image.shape, y_coords, x_coords)
            feature = base_net.predict(np.expand_dims(crop, axis=0))
            features.append(feature.reshape(feature.size))

    return np.array(features)