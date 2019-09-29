from types import GeneratorType

from object_counting.objects_counter.models.seq_sub import SeqSub
from object_counting.objects_counter.models.aso_sub import AsoSub
from object_counting.objects_counter.models.glance import Glance
from object_counting.objects_counter.models_e2e.aso_sub_fc_e2e import AsoSubFCE2E
from object_counting.objects_counter.models_e2e.seq_sub_e2e import SeqSubE2E
from object_counting.objects_counter.models_e2e.aso_sub_e2e import AsoSubE2E
from object_counting.objects_counter.models_e2e.glance_e2e import GlanceE2E


class ObjectsCounter:

    """The :class:`ObjectsCounter` object is the main interface of the Conting Objects with Deep Learning project. It provides the models to perform objects counting and exposes their main methods.

    Usually you create a :class:`ObjectsCounter` instance in your main module, and you define the :paramref:`objects_counter.ObjectsCounter.model_args` according to the chosen model. Available models are:

    - As defined in the paper:
        - :class:`object_counting.objects_counter.models.glance.Glance`
        - :class:`object_counting.objects_counter.models.aso_sub.AsoSub`
        - :class:`object_counting.objects_counter.models.seq_sub.SeqSub`
    - End-to-end implemetation
        - :class:`object_counting.objects_counter.models_e2e.glance_e2e.GlanceE2E`
        - :class:`object_counting.objects_counter.models_e2e.aso_sub_e2e.AsoSubE2E`
        - :class:`object_counting.objects_counter.models_e2e.seq_sub_e2e.SeqSubE2E`

    After having instantiated the model, it is possible to compile it, load weights, train it, evaluate it, and make predictions::

        glance = ObjectsCounter("glance",
                                ["person", "cat", "dog"],
                                {num_hidden=2, hidden_size=250},
                                input_shape=(224, 224, 3))

        # Train
        glance.compile()
        glance.train(train_data)

        # Evaluate
        glance.evaluate(test_dat)

        # Predict
        glance.predict(prediction_data)

    :param str count_model:  the name of the model to instantiate
    :param classes: a list consisting of the output classes
    :type classes: list of str
    :param dict model_args: a dict consisting of the parameter of the model. See specific model documentation.
    :param tuple input_shape: the shape of the input
    """

    def __init__(self, base_model, count_model, classes, model_args, input_shape=(224, 224, 3), e2e=False):

        self.input_shape = input_shape
        self.classes = classes
        self.model_name = count_model
        self.model_base = base_model
        self.e2e = e2e

        assert count_model in ['glance', 'asosub', 'asosubfc', 'seqsub', 'detect']

        self.model_args = model_args

        if count_model == 'glance' and not self.e2e:
            self.counting_model = Glance(input_shape, len(self.classes), **self.model_args)
        elif count_model == 'asosub' and not self.e2e:
            self.counting_model = AsoSub(input_shape, len(self.classes), **self.model_args)
        elif count_model == 'seqsub' and not self.e2e:
            self.counting_model = SeqSub(input_shape, len(self.classes), **self.model_args)
        elif count_model == 'glance' and self.e2e:
            self.counting_model = GlanceE2E(input_shape, self.model_base, len(self.classes), **self.model_args)
        elif count_model == 'asosub' and self.e2e:
            self.counting_model = AsoSubE2E(input_shape, self.model_base, len(self.classes), **self.model_args)
        elif count_model == 'asosubfc' and self.e2e:
            self.counting_model = AsoSubFCE2E(input_shape, self.model_base, len(self.classes), **self.model_args)
        elif count_model == 'seqsub' and self.e2e:
            self.counting_model = SeqSubE2E(input_shape, self.model_base, len(self.classes), **self.model_args)
        # elif count_model == 'detect':
        #     self.count_model = Detector(input_shape, len(self.classes))

    def compile(self, optimizer='adam', lr=0.001, loss='mse', summary=True):

        """Compile the Keras model with an optimizer, a loss function and a learning rate

        :param keras.Optimizer optimizer: optimizer used to perform error back-propagation
        :param float lr: learning rate
        :param str loss: name of the loss function. Possible values are "mse", "msle", and "mae"
        :param bool summary: whether to print or not the model summary.
        """

        self.counting_model.compile(optimizer, lr, loss, summary)

    def load_weights(self, path, by_name=False):

        """Load the model weights.

        :param str path: path to the weights file
        :param bool by_name: whether to match the weights by order or by layer name
        """

        self.counting_model.load_weights(path, by_name=by_name)

    def defreeze_layers(self, starting_idx):

        """Set all the layers to trainable starting from the starting_idx.

        :param starting_idx: index of the layer from which to defreeze the net
        """

        if self.e2e:
            self.counting_model.defreeze_layers(starting_idx)

    def train(self,
              X=None,
              y=None,
              train_gen=None,
              batch_size=None,
              epochs=1,
              initial_epoch=0,
              steps_per_epoch=None,
              val_data=None,
              validation_steps=None,
              lr_schedule=None,
              checkpoint_folder=None,
              verbose=1):

        """Train the :class:`ObjectsCounter`. Accepts either np.ndarrays or generators.
        Validation data is optional. It is possible to save the best weights providing a folder to store the .h5 file.

        :param np.ndarray X: samples
        :param np.ndarray y: labels
        :param generator train_gen: generator that return a tuple (samples_batch, labels_batch)
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

        if X is not None and y is not None:
            assert isinstance(val_data, tuple), "The validation dataset must be a tuple (samples, labels)"
            return self.counting_model.train(X, y, batch_size, epochs, initial_epoch, steps_per_epoch, val_data, validation_steps, lr_schedule, checkpoint_folder, verbose)
        elif train_gen is not None:
            assert isinstance(train_gen, GeneratorType), "The input must be a generator"
            assert isinstance(val_data, GeneratorType), "The validation input must be a generator"
            return self.counting_model.train_generator(train_gen, epochs, initial_epoch, steps_per_epoch, val_data, validation_steps, lr_schedule, checkpoint_folder, verbose)

    def evaluate(self,
                 X=None,
                 y=None,
                 gen=None,
                 batch_size=None,
                 steps=None):

        """Evaluate the :class:`ObjectsCounter`. Accepts either np.ndarrays or a generator.

        :param np.ndarray X: samples
        :param np.ndarray y: labels
        :param generator gen: generator that return a tuple (samples_batch, labels_batch)
        :param int batch_size: size of a batch
        :param int steps: number of test batches to process
        """

        if X is not None and y is not None:
            return self.counting_model.evaluate(X, y, batch_size, steps)
        elif gen is not None:
            assert isinstance(gen, GeneratorType), "The input must be a generator"
            return self.counting_model.evaluate_generator(gen, steps)

    def predict(self,
                X=None,
                y=None,
                gen=None,
                batch_size=None,
                steps=None):

        """Predict the count over images through the :class:`ObjectsCounter`. Accepts either a np.ndarray or a generator.

        :param np.ndarray X: samples
        :param generator gen: generator that return a tuple (samples_batch)
        :param int batch_size: size of a batch
        :param int steps: number of test batches to process
        """

        if X is not None:
            return self.counting_model.predict(X, batch_size, steps)
        elif gen is not None:
            assert isinstance(gen, GeneratorType), "The input must be a generator"
            return self.counting_model.predict_generator(gen, steps)












