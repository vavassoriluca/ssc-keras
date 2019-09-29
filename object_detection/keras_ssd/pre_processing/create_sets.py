import os
import random


def create_sets():

    train =open("../../dataset_new/sets/train.txt", 'w')
    validation = open("../../dataset_new/sets/validation.txt", 'w')
    trainval = open("../../dataset_new/sets/trainval.txt", 'w')
    test = open("../../dataset_new/sets/test.txt", 'w')
    all = open("../../dataset_new/sets/all.txt", 'w')

    split = 0.7

    files = os.listdir("../../dataset_new/images")
    random.shuffle(files)
    random.shuffle(files)
    random.shuffle(files)
    random.shuffle(files)

    for i, image in enumerate(files):

        if i < int(split*len(files)):
            train.write(image.replace(".jpg", "\n"))
            trainval.write(image.replace(".jpg", "\n"))

        elif i < int(len(files)*(((1-split)/2)+split)):
            validation.write(image.replace(".jpg", "\n"))
            trainval.write(image.replace(".jpg", "\n"))

        else:
            test.write(image.replace(".jpg", "\n"))

        all.write(image.replace(".jpg", "\n"))

    train.close()
    trainval.close()
    validation.close()
    test.close()
    all.close()


