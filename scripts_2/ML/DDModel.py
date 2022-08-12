"""
Version 1.1.2
"""
from .Tokenizer import DDTokenizer
from sklearn import preprocessing
from .DDModelExceptions import *
from tensorflow.keras import backend
from .Models import Models
from .Parser import Parser
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import time
import os

import warnings
warnings.filterwarnings('ignore')


class DDModel(Models):
    """
    A class responsible for creating, storing, and working with our deep docking models
    """

    def __init__(self, mode, input_shape, hyperparameters, metrics=None, loss='binary_crossentropy', regression=False,
                 name="model"):

        """
        Parameters
        ----------
        mode : str
            A string indicating which model to use
        input_shape : tuple or list
            The input shape for the model
        hyperparameters : dict
            A dictionary containing the hyperparameters for the DDModel's model
        metrics : list
            The metric(s) used by keras
        loss : str
            The loss function used by keras
        regression : bool
            Set to true if the model is performing regression
        """

        if metrics is None:
            self.metrics = ['accuracy']
        else:
            self.metrics = metrics

        # Use regression or use binary classification
        output_activation = 'linear' if regression else 'sigmoid'

        # choose the loss function
        self.loss_func = loss
        if regression and loss == 'binary_crossentropy':
            self.loss_func = 'mean_squared_error'
        hyperparameters["loss_func"] = self.loss_func

        if mode == "loaded_model":
            super().__init__(hyperparameters={'bin_array': [],
                                              'dropout_rate': 0,
                                              'learning_rate': 0,
                                              'num_units': 0,
                                              'epsilon': 0},
                             output_activation=output_activation, name=name)
            self.mode = ""

            self.input_shape = ()

            self.history = keras.callbacks.History()
            self.time = {"training_time": -1, "prediction_time": -1}
        else:
            # Create a model
            super().__init__(hyperparameters=hyperparameters,
                             output_activation=output_activation, name=name)

            self.mode = mode

            self.input_shape = input_shape

            self.history = keras.callbacks.History()
            self.time = {'training_time': -1, "prediction_time": -1}

            self.model = self._create_model()
            self._compile()

    def fit(self, train_x, train_y, epochs, batch_size, shuffle, class_weight, verbose, validation_data, callbacks):

        """
        Reshapes the input data and fits the model

        Parameters
        ----------
        train_x : ndarray
            Training data
        train_y : ndarray
            Training labels
        epochs : int
            Number of epochs to train on
        batch_size : int
            The batch size
        shuffle : bool
            Whether to shuffle the data
        class_weight : dict
            The class weights
        verbose : int
            The verbose
        validation_data : list
            The validation data and labels
        callbacks : list
            Keras callbacks
        """

        # First reshape the data to fit the chosen model
        # Here we form the shape
        shape_train_x = [train_x.shape[0]]
        shape_valid_x = [validation_data[0].shape[0]]
        for val in self.model.input_shape[1:]:
            shape_train_x.append(val)
            shape_valid_x.append(val)

        # Here we reshape the data
        # Format: shape = (size of data, input_shape[0], ..., input_shape[n]
        train_x = np.reshape(train_x, shape_train_x)
        validation_data_x = np.reshape(validation_data[0], shape_valid_x)
        validation_data_y = validation_data[1]
        validation_data = (validation_data_x, validation_data_y)

        # Track the training time
        training_time = time.time()

        # If we are in regression mode, ignore the class weights
        if self.output_activation == 'linear':
            class_weight = None

        # Train the model and store the history
        self.history = self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                                      class_weight=class_weight, verbose=verbose, validation_data=validation_data,
                                      callbacks=callbacks)

        # Store the training time
        training_time = time.time() - training_time
        self.time['training_time'] = training_time
        print("Training time:", training_time)

    def predict(self, x_test, verbose=0):

        """
        Reshapes the input data and returns the models predictions

        Parameters
        ----------
        x_test : ndarray
            The test data

        verbose : int
            The verbose of the model's prediction

        Returns
        -------
        predictions : ndarray
            The model's predictions
        """
        # We must reshape the test data to fit our model
        shape = [x_test.shape[0]]
        for val in list(self.model.input_shape)[1:]:
            shape.append(val)

        x_test = np.reshape(x_test, newshape=shape)

        # Predict and return the predictions
        prediction_time = time.time()  # Keep track of how long prediction took
        predictions = self.model.predict(x_test, verbose=verbose)  # Predict
        prediction_time = time.time() - prediction_time  # Update prediction time
        self.time['prediction_time'] = prediction_time  # Store the prediction time

        return predictions

    def save(self, path, json=False):
        self._write_stats_to_file(path)
        if json:
            json_model = self.model.to_json()
            with open(path + ".json", 'w') as json_file:
                json_file.write(json_model)
        else:
            try:
                self.model.save(path, save_format='h5')
            except:
                print("Could not save as h5 file. This is probably due to tensorflow version.")
                print("If the model is saved a directory, it will cause issues.")
                print("Trying to save again...")
                self.model.save(path)

    def load_stats(self, path):
        """
        Load the stats from a .ddss file into the current DDModel

        Parameters
        ----------
        path : str
        """

        info = Parser.parse_ddss(path)
        for key in info.keys():
            try:
                self.__dict__[key] = info[key]
            except KeyError:
                print(key, 'is not an attribute of this class.')
        self.input_shape = "Loaded Model -> Input shape will be inferred"

        if self.time == {}:
            self.time = {"training_time": 'Could Not Be Loaded', "prediction_time": 'Could Not Be Loaded'}

    def _write_stats_to_file(self, path="", return_string=False):
        info = "* {}'s Stats * \n".format(self.name)
        info += "- Model mode: " + self.mode + " \n"
        info += "\n"
        # Write the timings
        if isinstance(self.time['training_time'], str) == False and self.time['training_time'] > -1:
            if isinstance(self.history, dict):
                num_eps = self.history['total_epochs']
            else:
                num_eps = len(self.history.history['loss'])

            info += "- Model Time: \n"
            info += "   - training_time: {train_time}".format(train_time=self.time['training_time']) + "  \n"
            info += "   - time_per_epoch: {epoch_time}".format(epoch_time=(self.time['training_time'] / num_eps)) + " \n"
            info += "   - prediction_time: {pred_time}".format(pred_time=self.time['prediction_time']) + " \n"
        else:
            info += "- Model Time: \n"
            info += "   - Model has not been trained yet. \n"
        info += "\n"

        # Write the history
        try:
            info += "- History Stats: \n"
            if isinstance(self.history, dict):
                hist = self.history
            else:
                hist = self.history.history

            # Get all the history values and keys stores
            for key in hist:
                try:
                    info += "   - {key}: {val} \n".format(key=key, val=hist[key][-1])
                except TypeError:
                    info += "   - {key}: {val} \n".format(key=key, val=hist[key])

            try:
                try:
                    info += "   - total_epochs: {epochs}".format(epochs=len(hist['loss']))
                except TypeError:
                    pass

                info += "\n"
            except KeyError:
                info += "   - Model has not been trained yet. \n"

        except AttributeError or KeyError:
            # Get all the history values and keys stores
            info += "   - Model has not been trained yet. \n"
        info += "\n"

        # Write the hyperparameters
        info += "- Hyperparameter Stats: \n"
        for key in self.hyperparameters.keys():
            if key != 'bin_array' or len(self.hyperparameters[key]) > 0:
                info += "   - {key}: {val} \n".format(key=key, val=self.hyperparameters[key])
        info += "\n"

        # Write stats about the model architecture
        info += "- Model Architecture Stats: \n"
        try:
            trainable_count = int(
                np.sum([backend.count_params(p) for p in set(self.model.trainable_weights)]))
            non_trainable_count = int(
                np.sum([backend.count_params(p) for p in set(self.model.non_trainable_weights)]))

            info += '   - total_params: {:,} \n'.format(trainable_count + non_trainable_count)
            info += '   - trainable_params: {:,}  \n'.format(trainable_count)
            info += '   - non_trainable_params: {:,}  \n'.format(non_trainable_count)
            info += "\n"
        except TypeError or AttributeError:
            info += '   - total_params: Cannot be determined \n'
            info += '   - trainable_params: Cannot be determined \n'
            info += '   - non_trainable_params: Cannot be determined \n'
            info += "\n"

        # Create a layer display
        display_string = ""
        for i, layer in enumerate(self.model.layers):
            if i == 0:
                display_string += "Input: \n"

            display_string += "     [ {name} ] \n".format(name=layer.name)
        info += display_string

        if not return_string:
            with open(path + '.ddss', 'w') as stat_file:
                stat_file.write(info)
        else:
            return info

    def _create_model(self):
        """Creates and returns a model

        Raises
        ------
        IncorrectModelModeError
            If a mode was passed that does not exists this error will be raised
        """
        # Try creating the model and if failed raise exception
        try:
            model = getattr(self, self.mode, None)(self.input_shape)
        except TypeError:
            raise IncorrectModelModeError(self.mode, Models.get_available_modes())
        return model

    def _compile(self):
        """Compiles the DDModel object's model"""

        if 'epsilon' not in self.hyperparameters.keys():
            self.hyperparameters['epsilon'] = 1e-06

        adam_opt = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate'],
                                            epsilon=self.hyperparameters['epsilon'])
        self.model.compile(optimizer=adam_opt, loss=self.loss_func, metrics=self.metrics)

    @staticmethod
    def load(model, **kwargs):
        pre_compiled = True
        # Can be a path to a model or a model instance
        if type(model) is str:
            dd_model = DDModel(mode="loaded_model", input_shape=[], hyperparameters={})

            # If we would like to load from a json, we can do that as well.
            if '.json' in model:
                dd_model.model = tf.keras.models.model_from_json(open(model).read(),
                                                                 custom_objects=Models.get_custom_objects())
                model = model.replace('.json', "")
                pre_compiled = False
            else:
                dd_model.model = tf.keras.models.load_model(model, custom_objects=Models.get_custom_objects())
        else:
            dd_model = DDModel(mode="loaded_model", input_shape=[], hyperparameters={})
            dd_model.model = model

        if 'kt_hyperparameters' in kwargs.keys():
            hyp = kwargs['kt_hyperparameters'].get_config()['values']
            for key in hyp.keys():
                try:
                    dd_model.__dict__['hyperparameters'][key] = hyp[key]
                    if key == 'kernel_reg':
                        dd_model.__dict__['hyperparameters'][key] = ['None', 'Lasso', 'l1', 'l2'][int(hyp[key])]

                except KeyError:
                    print(key, 'is not an attribute of this class.')
        else:
            # Try to load a stats file
            try:
                dd_model.load_stats(model + ".ddss")
            except TypeError or FileNotFoundError:
                print("Could not find a stats file...")

        if 'metrics' in kwargs.keys():
            dd_model.metrics = kwargs['metrics']
        else:
            dd_model.metrics = ['accuracy']

        if not pre_compiled:
            dd_model._compile()

        if 'name' in kwargs.keys():
            dd_model.name = kwargs['name']

        dd_model.mode = 'loaded_model'
        return dd_model

    @staticmethod
    def process_smiles(smiles, vocab_size=100, fit_range=1000, normalize=True, use_padding=True, padding_size=None, one_hot=False):
        # Create the tokenizer
        tokenizer = DDTokenizer(vocab_size)
        # Fit the tokenizer
        tokenizer.fit(smiles[0:fit_range])
        # Encode the smiles
        encoded_smiles = tokenizer.encode(data=smiles, use_padding=use_padding,
                                          padding_size=padding_size, normalize=normalize)

        if one_hot:
            encoded_smiles = DDModel.one_hot_encode(encoded_smiles, len(tokenizer.word_index))

        return encoded_smiles

    @staticmethod
    def one_hot_encode(encoded_smiles, unique_category_count):
        one_hot = keras.backend.one_hot(encoded_smiles, unique_category_count)
        return one_hot

    @staticmethod
    def normalize(values: pd.Series):
        assert type(values) is pd.Series, "Type Error -> Expected pandas.Series"
        # Extract the indices and name
        indices = values.index
        name = values.index.name

        # Normalizes values
        normalized_values = preprocessing.minmax_scale(values, (0, 1))

        # Create a pandas series to return
        values = pd.Series(index=indices, data=normalized_values, name=name)
        return values

    def __repr__(self):
        return self._write_stats_to_file(return_string=True)
