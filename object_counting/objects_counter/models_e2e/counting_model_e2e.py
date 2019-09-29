import os

from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_squared_logarithmic_error

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, CSVLogger
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ModelCheckpoint


class CountingModelE2E:

    """The :class:`CountingModelE2E` object is the core of the Conting Objects with Deep Learning project.
    It is the parent class of all the models and provides their main methods.

   :param tuple input_shape: the shape of the input
   :param int num_classes: the number output classes
   """

    def __init__(self, input_shape=(224,224,3), base_model='inceptionresnet', num_classes=20):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.base_model_name = base_model
        self.base_model = None

    def compile(self, optimizer='adam', lr=0.001, loss='mse', summary=True):

        """Compile the Keras model with an optimizer, a loss function and a learning rate

        :param keras.Optimizer optimizer: optimizer used to perform error back-propagation
        :param float lr: learning rate
        :param str loss: name of the loss function. Possible values are "mse", "msle", and "mae"
        :param bool summary: whether to print or not the model summary.
        """

        if optimizer == 'adam':
            count_optimizer = Adam(lr=lr)
        elif optimizer == 'sgd':
            count_optimizer = SGD(lr=lr)
        else:
            raise ValueError("Optimizer not valid.")

        if loss == 'mse':
            count_loss = mean_squared_error
        elif loss == 'msle':
            count_loss = mean_squared_logarithmic_error
        elif loss == 'mae':
            count_loss = mean_absolute_error
        else:
            raise ValueError("Loss function not valid.")

        model = self.model
        model.compile(count_optimizer, loss=count_loss)

        if summary:
            self.model.summary()

    def load_weights(self, path, by_name=False):

        """Load the model weights.

        :param str path: path to the weights file
        :param bool by_name: whether to match the weights by order or by layer name
        """

        self.model.load_weights(path, by_name=by_name)

    def defreeze_layers(self, starting_idx):

        """Set all the layers to trainable starting from the starting_idx.

        :param starting_idx: index of the layer from which to defreeze the net
        """

        for l in self.model.layers[starting_idx:]:
            l.trainable = True

    def train(self,
              X=None,
              y=None,
              batch_size=None,
              epochs=1,
              initial_epoch=0,
              steps_per_epoch=None,
              val_data=None,
              validation_steps=None,
              lr_schedule=None,
              checkpoint_folder=None,
              verbose=1):

        """Train the :class:`ObjectsCounter`. Data are provided as np.ndarrays, the first dimension is the number of samples.
        Validation data is optional. It is possible to save the best weights providing a folder to store the .h5 file.

        :param np.ndarray X: samples
        :param np.ndarray y: labels
        :param int batch_size: size of a batch
        :param int epochs: total number of training iterations
        :param int initial_epoch: first epoch
        :param int steps_per_epoch: number of training batches to process in one epoch
        :param tuple val_data: validation data as a tuple (X_val, y_val)
        :param int validation_steps: only relevant if :code:`steps_per_epoch` is set. Number of validation batches to process in the validation phase.
        :param function lr_schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning rate and returns a new learning rate as output (float).
        :param str checkpoint_folder: folder where to save the model checkpoints
        :param int verbose: whether to print or not progress bars, either 1 or 0

        """

        # CREATE CALLBACKS

        callbacks = []

        if lr_schedule is not None:
            callbacks.append(LearningRateScheduler(lr_schedule))

        if checkpoint_folder is not None:
            checkpoint_folder = os.path.join(checkpoint_folder)
            callbacks.append(ModelCheckpoint(filepath=checkpoint_folder+type(self).__name__+'_weights_epoch-{epoch:04d}_val_loss-{val_loss:.4f}.h5',
                                             monitor='val_loss', save_best_only=True, save_weights_only=True, mode='auto'))
            callbacks.append(CSVLogger(filename='{}{}_training_log.csv'.format(checkpoint_folder, type(self).__name__),
                                       separator=',', append=True))

        callbacks.append(TerminateOnNaN())

        return self.model.fit(x=X,
                              y=y,
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_data=val_data,
                              shuffle=True,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              verbose=verbose)

    def evaluate(self,
                 X=None,
                 y=None,
                 batch_size=None,
                 steps=None):

        """Evaluate the :class:`ObjectsCounter`. Data are provided as np.ndarrays, the first dimension is the number of samples.

        :param np.ndarray X: samples
        :param np.ndarray y: labels
        :param int batch_size: size of a batch
        :param int steps: number of test batches to process
        """

        return self.model.evaluate(x=X, y=y, batch_size=batch_size, steps=steps)

    def predict(self,
                X,
                batch_size=None,
                steps=None):

        """Predict the count over images through the :class:`ObjectsCounter`. Data are provided as np.ndarrays, the first dimension is the number of samples.

        :param np.ndarray X: samples
        :param int batch_size: size of a batch
        :param int steps: number of batches to process
        """

        return self.model.predict(X, batch_size=batch_size, steps=steps).clip(min=0)

    def train_generator(self,
                        train_gen,
                        epochs=1,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        val_gen=None,
                        validation_steps=None,
                        lr_schedule=None,
                        checkpoint_folder=None,
                        verbose=1):

        """Train the :class:`objects_counter.ObjectsCounter` feeding data through a :class:`utils.data_generator.DataGenerator`.
        Validation data is optional. It is possible to save the best weights providing a folder to store the .h5 file.

        :param generator train_gen: generator that return a tuple (samples_batch, labels_batch)
        :param int epochs: total number of training iterations
        :param int initial_epoch: first epoch
        :param int steps_per_epoch: number of training batches to process in one epoch
        :param generator val_gen: generator that return a tuple (samples_batch, labels_batch)
        :param int validation_steps: number of validation batches to process at the end of an epoch
        :param function lr_schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning rate and returns a new learning rate as output (float).
        :param str checkpoint_folder: folder where to save the model checkpoints
        :param int verbose: whether to print or not progress bars, either 1 or 0

        """

        # CREATE CALLBACKS

        callbacks = []

        if lr_schedule is not None:
            callbacks.append(LearningRateScheduler(lr_schedule))

        if checkpoint_folder is not None:
            checkpoint_folder = os.path.join(checkpoint_folder)
            callbacks.append(ModelCheckpoint(filepath=checkpoint_folder+type(self).__name__+'_weights_epoch-{epoch:04d}_val_loss-{val_loss:.4f}.h5',
                                             monitor='val_loss', save_best_only=True, save_weights_only=True, mode='auto'))
            callbacks.append(CSVLogger(filename='{}{}_training_log.csv'.format(checkpoint_folder, type(self).__name__),
                                       separator=',', append=True))

        callbacks.append(TerminateOnNaN())

        return self.model.fit_generator(train_gen,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=val_gen,
                                        validation_steps=validation_steps,
                                        use_multiprocessing=False,
                                        shuffle=True,
                                        initial_epoch=initial_epoch)

    def evaluate_generator(self,
                           data,
                           steps=None):

        """Evaluate the :class:`objects_counter.ObjectsCounter` feeding data through a :class:`utils.data_generator.DataGenerator`.

        :param generator data: generator that return a tuple (samples_batch, labels_batch)
        :param int steps: number of test batches to process
        """

        return self.model.evaluate_generator(data, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    def predict_generator(self,
                          data,
                          steps=None):

        """Predict the count over images through the :class:`objects_counter.ObjectsCounter`. Data are provided by a :class:`utils.data_generator.DataGenerator`.

        :param generator data: generator that return a tuple (samples_batch)
        :param int steps: number of batches to process
        """

        return self.model.predict_generator(data, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1).clip(min=0)

