# import the necessary packages
import cv2
import os
import sys
import random

path = os.path.join(sys.argv[1], '')

for imagename in os.listdir(path):

    # load the image and show it
    image = cv2.imread(path + imagename)

    # ROTATION

    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    for i in range(1, 9):
        # rotate the image by 20*i degrees
        M = cv2.getRotationMatrix2D(center, 20*i + random.randint(-5, 5), 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(path + imagename.replace('.jpg', '') + "{:04d}".format(i*20) + ".jpg", rotated)

