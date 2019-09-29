import os
import numpy as np
from matplotlib import pyplot as plt

from object_detection.keras_ssd.data_generator.object_detection_2d_data_generator import DataGenerator
from object_detection.keras_ssd.data_generator.object_detection_2d_geometric_ops import Resize
from object_detection.keras_ssd.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
import random


class ImagePlotter(object):

    def __init__(self, images_dir, annotations_dir, set_path, classes=None):

        self.images_dir = os.path.join(images_dir)
        self.annotations_dir = os.path.join(annotations_dir)
        self.set_path = set_path
        self.classes = classes
        if classes is None:
            self.classes = classes = ['background',
                                      'screw_0', 'screw_1', 'screw_2', 'screw_3',
                                      'screw_4', 'screw_5', 'screw_6', 'screw_7',
                                      'screw_8', 'screw_9']
        self.dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        self.dataset.parse_xml(images_dirs=[self.images_dir],
                        image_set_filenames=[self.set_path],
                        annotations_dirs=[self.annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

    def plot_all(self):

        # For the generator:
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=300, width=300)

        generator = self.dataset.generate(batch_size=1,
                              shuffle=True,
                              transformations=[convert_to_3_channels,
                                               resize],
                              label_encoder=None,
                              returns={'processed_images',
                                       'filenames',
                                       'inverse_transform',
                                       'original_images',
                                       'original_labels'},
                              keep_images_without_gt=False)

        for batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels in generator:

            # Set the colors for the bounding boxes
            colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes))).tolist()

            plt.figure(figsize=(20, 12))
            plt.imshow(batch_original_images[0])

            current_axis = plt.gca()

            for box in batch_original_labels[0]:
                xmin = box[1]
                ymin = box[2]
                xmax = box[3]
                ymax = box[4]
                label = '{}'.format(self.classes[int(box[0])])
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=colors[random.randint(0,10)], fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                  bbox={'facecolor': 'green', 'alpha': 1.0})

            plt.show()

