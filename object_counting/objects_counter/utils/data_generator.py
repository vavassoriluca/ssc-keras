import h5py
import random
import numpy as np
import os
import cv2

from tqdm import tqdm

from object_counting.objects_counter.utils.feature_extractor import FeatureExtractor
from object_counting.objects_counter.gt_generation.gt_gen import GTGen
from object_counting.objects_counter.gt_generation.gt_gen_e2e import GTGenE2E


class DataGenerator:

    """The :class:`DataGenerator` provides methods to load the dataset into memory from disk or hdf5 datasets.
    It allows to save them to hdf5 datasets as well. Moreover, it instantiates generators to fit and evaluate the models.

    :param classes: the names of the classes
    :type classes: list of str
    :param str count_model: the type of count model
    :param str base_model: the type of CNN Classifier to extract the features
    :param str base_model_weights: pth to the weights
    :param tuple input_shape: required input shape of the base model
    :param tuple grid_division: division of the image in grids, None otherwise'
    :param dict gt_mode: args to define the mode to split the count into cells
    """

    def __init__(self,
                 classes,
                 count_model='glance',
                 base_model='vgg16',
                 base_model_weights=None,
                 input_shape=(224, 224, 3),
                 grid_division=None,
                 gt_mode={'mode': 'linear'}):

        assert count_model in ['glance', 'asosub', 'asosubfc', 'seqsub', 'seqsub7', 'detect'], "Unknown count model"
        assert base_model in ['vgg16', 'mobilenet', 'resnet50', 'resnet152', 'inceptionresnet'], "Unknown base model"
        if grid_division is None:
            assert count_model not in ['asosub', 'seqsub'], "A model requiring grids is provided, but no grid division has been given"
        else:
            assert count_model not in ['glance','detect'], "A model not requiring grids is provided, but grid division has been given"

        self.classes = classes
        self.count_model = count_model
        self.base_model = base_model
        self.input_shape = input_shape
        self.grid_division = grid_division
        self.feature_ext = FeatureExtractor(base_model, self.input_shape, base_weights=base_model_weights)
        self.dataset_size = 0
        self.gt_mode = gt_mode

    def load_dataset(self,
                     sets=None,
                     images_folder=None,
                     annotations_folder=None,
                     annotations_ext='xml',
                     setnames=None,
                     h5dataset_path=None,
                     h5saving_folder=None,
                     h5saving_file=None,
                     load_into_memory=True,
                     serialize_cells=False,
                     gt_only=False,
                     data_augmentation_pipeline={}):

        """This method load a dataset from an hdf5 file or from folders containing the data. It directly process the images to extract features to feed the count model.

        :param sets: list of strings, names of the .txt files containing the name of the images to include in the set (name without extension)
        :type sets: list of str
        :param str images_folder: path to the folder containing all the images
        :param str annotations_folder: path to the folder containing all the annotation files, the name must match the one of the .jpg file
        :param str annotations_ext: extension of the annotation files (e.g xml)
        :param setnames: name of the file where to save the hdf5 datasets
        :type setnames: list of str
        :param str h5dataset_path: path to the hdf5 for the dataset to be loaded, set to None if loading images from folders
        :param str h5saving_folder: path to the folder where to save hdf5, if set to None, no file will be saved
        :param str h5saving_file: in case of loading a dataset fro man hdf5, the saving path must be specified here, otherwise set to None
        :param bool load_into_memory: set to True if the data must be load into memory
        :param bool serialize_cells: set to true if the grid division ,ust be serialized so that each cell is an independent element
        :param bool gt_only: whether to output only the gt or not
        :param dict data_augmentation_pipeline: list of transformations with relative parameters. To be implemented

        :return: a list containing the datasets
        :rtype: list of np.ndarrays
        """

        assert (serialize_cells and self.count_model == 'asosub') or (not serialize_cells), "Cells are serializable only for Associative Subitizing"
        assert not gt_only or (gt_only and annotations_folder is not None)
        assert (not serialize_cells) or (serialize_cells and self.grid_division is not None)


        h5 = None
        if h5saving_file is not None:
            h5 = h5py.File(h5saving_file, 'w')
        if load_into_memory:
            X_datasets = [] # store here the features if loa_into_memory is True
            y_datasets = [] # store here the features if loa_into_memory is True

        if annotations_folder is not None:
            self.gt_generator = GTGen

        # If the hdf5 is provided, load the data from it
        if h5dataset_path is not None:
            h5_dataset = h5py.File(h5dataset_path, 'r')

            t = tqdm(h5_dataset.keys())
            t.set_description_str(desc="Loading dataset and computing features", refresh=True)

            # iterate over the images stored in the dataset
            for k in t:
                features = self.feature_ext.extract(h5_dataset[k]['img'], grid_division=self.grid_division)
                if load_into_memory:
                    if serialize_cells:
                        X_datasets.extend(features)
                    else:
                        X_datasets.append(features)
                if h5saving_file is not None:
                    h5.create_dataset('{}/features'.format(k), data=features)
                if h5_dataset[k]['gt'] is not None:
                    if load_into_memory:
                        if serialize_cells:
                            y_datasets.extend(h5_dataset[k]['gt'])
                        else:
                            y_datasets.append(h5_dataset[k]['gt'])
                    if h5saving_file is not None:
                        h5.create_dataset('{}/gt'.format(k), h5_dataset[k]['gt'])

            # necessary to write the data on the h5 file
            if h5saving_file is not None:
                h5.flush()
                h5.close()

            if load_into_memory:
                return X_datasets, y_datasets

        # if images are provided through folders and sets
        elif images_folder is not None:

            assert len(sets) == len(setnames)

            # iterate over the sets, the output will be one per set
            for i in range(len(sets)):

                if load_into_memory:
                    ftrs = []
                    gts = []
                if h5saving_folder is not None:
                    h5=h5py.File(os.path.join(h5saving_folder, '{}.h5'.format(setnames[i])), 'w')

                t = tqdm(open(sets[i], 'r').read().splitlines())
                t.set_description_str(desc="Loading dataset and computing features", refresh=True)

                #iterate over each element of the given set, compute the feature and save that into memory and/or in a h5 file
                for filename in t:

                    ftr, gt = self.load_img_ann(filename, images_folder, annotations_folder, annotations_ext, gt_only)

                    if load_into_memory:
                        if serialize_cells:
                            if not gt_only:
                                ftrs.extend(ftr)
                            if gt is not None:
                                gts.extend(gt)
                        else:
                            if not gt_only:
                                ftrs.append(ftr)
                            if gt is not None:
                                gts.append(gt)

                    if h5saving_folder is not None:

                        if serialize_cells:
                            for f in range(ftr.shape[0]):
                                if not gt_only:
                                    h5.create_dataset('{}{:02d}/features'.format(filename, f), data=ftr[f])
                                if gt is not None:
                                    h5.create_dataset('{}{:02d}/gt'.format(filename, f), data=gt[f])
                        else:
                            if not gt_only:
                                h5.create_dataset('{}/features'.format(filename), data=ftr)
                            if gt is not None:
                                h5.create_dataset('{}/gt'.format(filename), data=gt)
                        h5.flush()

                if load_into_memory:
                    if not gt_only:
                        X_datasets.append(np.array(ftrs))
                    if annotations_folder is not None:
                        y_datasets.append(np.array(gts))

                if h5saving_folder is not None:
                    h5.flush()
                    h5.close()

            if load_into_memory:
                return X_datasets, y_datasets

    def load_dataset_noftr(self,
                           sets=None,
                           images_folder=None,
                           annotations_folder=None,
                           annotations_ext='xml',
                           setnames=None,
                           h5saving_folder=None,
                           h5saving_file=None,
                           load_into_memory=True,
                           serialize_cells=False,
                           gt_only=False,
                           data_augmentation_pipeline={}):

        """This method load a dataset from an hdf5 file or from folders containing the data. It doesn't process the images, it loads them as they are..

        :param sets: list of strings, names of the .txt files containing the name of the images to include in the set (name without extension)
        :type sets: list of str
        :param str images_folder: path to the folder containing all the images
        :param str annotations_folder: path to the folder containing all the annotation files, the name must match the one of the .jpg file
        :param str annotations_ext: extension of the annotation files (e.g xml)
        :param setnames: name of the file where to save the hdf5 datasets
        :type setnames: list of str
        :param str h5dataset_path: path to the hdf5 for the dataset to be loaded, set to None if loading images from folders
        :param str h5saving_folder: path to the folder where to save hdf5, if set to None, no file will be saved
        :param str h5saving_file: in case of loading a dataset fro man hdf5, the saving path must be specified here, otherwise set to None
        :param bool load_into_memory: set to True if the data must be load into memory
        :param bool serialize_cells: whether to serialize the cells of the image or not
        :param bool gt_only: whether to output only the gt or not
        :param dict data_augmentation_pipeline: list of transformations with relative parameters. To be implemented

        :return: a list containing the datasets
        :rtype: list of np.ndarrays
        """

        assert not gt_only or (gt_only and annotations_folder is not None)
        assert (serialize_cells and self.count_model == 'asosub') or (not serialize_cells),\
            "Cells are serializable only for Associative Subitizing"
        assert (not serialize_cells) or (serialize_cells and self.grid_division is not None)

        h5 = None
        if h5saving_file is not None:
            h5 = h5py.File(h5saving_file, 'w')
        if load_into_memory:
            X_datasets = [] # store here the features if load_into_memory is True
            y_datasets = [] # store here the features if load_into_memory is True

        if annotations_folder is not None:
            if 'asosub' in self.count_model:
                self.gt_generator = GTGenE2E
            else:
                self.gt_generator = GTGen

        assert len(sets) == len(setnames)

        # iterate over the sets, the output will be one per set
        for i in range(len(sets)):

            if load_into_memory:
                ftrs = []
                gts = []
            if h5saving_folder is not None:
                h5=h5py.File(os.path.join(h5saving_folder, '{}.h5'.format(setnames[i])), 'w')

            t = tqdm(open(sets[i], 'r').read().splitlines())
            t.set_description_str(desc="Loading dataset and converting bounding boxes", refresh=True)

            #iterate over each element of the given set, compute the feature and save that into memory and/or in a h5 file
            for line in t:
                filename = line.split(' ')[0]
                ftr, gt = self.load_img_ann_noftr(filename, images_folder, annotations_folder, annotations_ext, gt_only, serialize_cells)
                if load_into_memory:
                    if serialize_cells:
                        if not gt_only:
                            ftrs.extend(ftr.tolist())
                        if gt is not None:
                            gts.extend(gt)
                    else:
                        if not gt_only:
                            ftrs.append(ftr.tolist())
                        if gt is not None:
                            gts.append(gt)
                if h5saving_folder is not None:
                    if serialize_cells:
                        for f, ff in enumerate(ftr.tolist()):
                            if not gt_only:
                                h5.create_dataset('{}{:02d}/features'.format(filename, f), data=ff)
                            if gt is not None:
                                h5.create_dataset('{}{:02d}/gt'.format(filename, f), data=gt[f])
                    else:
                        if not gt_only:
                            h5.create_dataset('{}/features'.format(filename), data=ftr)
                        if gt is not None:
                            h5.create_dataset('{}/gt'.format(filename), data=gt)

            if load_into_memory:
                if not gt_only:
                    X_datasets.append(np.array(ftrs))
                if annotations_folder is not None:
                    y_datasets.append(np.array(gts))

            if h5saving_folder is not None:
                h5.flush()
                h5.close()

        if load_into_memory:
            return X_datasets, y_datasets

    def load_img_ann(self, filename, images_folder, annotations_folder, ann_ext, gt_only):

        """This method load an image from the path and extract the features using the base_net.

        :param str filename: name of the image
        :param str images_folder: path to the image folder
        :param str annotations_folder: path to the annotation folder
        :param str ann_ext: extension of the annotation file
        :param bool gt_only: whether to output only the gt or not

        :returns: features extracted[, ground truth related]
        :rtype: np.ndarray
        """

        imgpath = os.path.join(images_folder, '{}.jpg'.format(filename))
        annpath = None
        if annotations_folder is not None:
            annpath = os.path.join(annotations_folder, '{}.{}'.format(filename, ann_ext))
        image = cv2.imread(imgpath)
        # TODO DATA AUGMENTATION
        image_shape = image.shape[:2]
        if not gt_only:
            if self.grid_division is None:
                image = cv2.resize(image, self.input_shape[:2])
            features = self.feature_ext.extract(image, grid_division=self.grid_division)
            if annpath is not None:
                gt = self.gt_generator.get_gt(self.classes, annpath, annotations_ext=ann_ext, input_shape=image_shape,
                                              grid_division=self.grid_division, **self.gt_mode)

                return features, gt
            else:
                return features, None
        else:
            gt = self.gt_generator.get_gt(self.classes, annpath, annotations_ext=ann_ext, input_shape=image_shape,
                                          grid_division=self.grid_division, **self.gt_mode)
            return None, gt

    def load_img_ann_noftr(self, filename, images_folder, annotations_folder, ann_ext, gt_only, serialize_cells):

        """This method load an image from the path and extract the features using the base_net.

        :param str filename: name of the image
        :param str images_folder: path to the image folder
        :param str annotations_folder: path to the annotation folder
        :param str ann_ext: extension of the annotation file
        :param bool gt_only: whether to output only the gt or not

        :returns: features extracted[, ground truth related]
        :rtype: np.ndarray
        """

        imgpath = os.path.join(images_folder, '{}.jpg'.format(filename))
        annpath = None
        if annotations_folder is not None:
            annpath = os.path.join(annotations_folder, '{}.{}'.format(filename, ann_ext))
        image = cv2.imread(imgpath)
        # TODO DATA AUGMENTATION
        image_shape = image.shape[:2]
        image = cv2.resize(image, self.input_shape[:2])
        if not gt_only:
            if annpath is not None:
                gt = self.gt_generator.get_gt(self.classes, annpath, annotations_ext=ann_ext, input_shape=image_shape,
                                              grid_division=self.grid_division, **self.gt_mode)
                if serialize_cells:
                    return self.split_cells(image), np.reshape(np.transpose(gt, (1, 0, 2)), (-1, len(self.classes)))
                else:
                    return image, gt
            else:
                if serialize_cells:
                    return self.split_cells(), None
                else:
                    return image, None
        else:
            gt = self.gt_generator.get_gt(self.classes, annpath, annotations_ext=ann_ext, input_shape=image_shape,
                                          grid_division=self.grid_division, **self.gt_mode)
            if serialize_cells:
                return None, np.reshape(np.transpose(gt, (1, 0, 2)), (-1, len(self.classes)))
            else:
                return None, gt

    def split_cells(self, image):

        """ Take an image and split it into cells

        :param np.ndarray image: the image to split into cells
        :return: list of np.ndarrays
        """

        cells = []
        x_coords = np.linspace(0, image.shape[1], self.grid_division[0] + 1)
        y_coords = np.linspace(0, image.shape[0], self.grid_division[1] + 1)
        for i in range(self.grid_division[0]):
            for j in range(self.grid_division[1]):
                crop = image[int(y_coords[j]):int(y_coords[j + 1]), int(x_coords[i]):int(x_coords[i + 1]), :].copy()
                cells.append(cv2.resize(crop, self.input_shape[:2]))
        return np.array(cells)

    def h5features_to_memory(self, h5_path, gt=True):

        """This method load a dataset of features stored in a .h5 file into memory.

        :param str h5_path: path to the h5 dataset with features (and gt optionally)
        :param bool gt: whether labels are provided or not

        :returns: samples[, labels]
        :rtype: np.ndarray
        """

        hdf5_dataset = h5py.File(h5_path, 'r')

        X = []
        y = []

        t = tqdm(hdf5_dataset.keys())
        t.set_description_str(desc="Loading dataset from h5", refresh=True)

        for k in t:
            X.append(hdf5_dataset[k]['features'].value)
            if hdf5_dataset[k]['gt'] is not None:
                y.append(hdf5_dataset[k]['gt'].value)

        hdf5_dataset.close()

        if gt:
            return np.array(X), np.array(y)
        else:
            return np.array(X)


    def generate(self, X=None, y=None, h5_path=None, batch_size=9, shuffle=True):

        """This method instantiate a generator from either a list(s) of elements or a hdf5 dataset. Ground truth is optional.
        It is possible to provide both the list(s) or the hdf5, the in-memory dataset will be preferred.

        :param np.ndarray X: samples
        :param np.ndarray y: labels
        :param str h5_path: path to the hdf5 dataset
        :param int batch_size: number of samples per batch
        :param bool shuffle: whether to shuffle the dataset or not

        :returns: batch of samples (and ground truth if provided)
        :rtype: generator
        """


        #############################################################################################
        # Shuffle the dataset if necessary
        #############################################################################################

        hdf5_dataset = None
        if h5_path is not None:
            hdf5_dataset = h5py.File(h5_path, 'r')
            # self.dataset_size = hdf5_dataset.attrs['dataset_size']
            dataset_indices = list(hdf5_dataset.keys())
            self.dataset_size = len(dataset_indices)

        elif X is not None:
            self.dataset_size = X.shape[0]
            dataset_indices = list(range(X.shape[0]))
        else:
            raise ValueError("No data provided")

        if shuffle:
            random.shuffle(dataset_indices)

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = current - self.dataset_size

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

                if shuffle:
                    random.shuffle(dataset_indices)

            #########################################################################################
            # Get the images, (maybe) labels, for this batch.
            #########################################################################################
            batch_indices = dataset_indices[current:current+batch_size]
            if X is not None:
                for i in batch_indices:
                    batch_X.append(X[i, ...])
                    if y is not None:
                        batch_y.append(y[i])

            elif hdf5_dataset is not None:
                for i in batch_indices:
                    batch_X.append(hdf5_dataset[i]['features'].value)
                    if hdf5_dataset[i]['gt'] is not None:
                        batch_y.append(hdf5_dataset[i]['gt'].value)

            current += batch_size

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.

            batch_X = np.array(batch_X)
            if len(batch_y) > 0:
                batch_y = np.array(batch_y)
                yield [batch_X, batch_y]
            else:
                del batch_y
                yield [batch_X]





