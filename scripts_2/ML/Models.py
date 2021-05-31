"""
Version 1.1.0

The model to be used in deep docking
James Gleave
"""

import keras
from ML.lasso_regularizer import Lasso
from ML.DDMetrics import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Activation, BatchNormalization, Dropout, LSTM,
                          Conv2D, MaxPool2D, Flatten, Embedding, MaxPooling1D,
                          Conv1D)
from tensorflow.keras.regularizers import *

import warnings
warnings.filterwarnings('ignore')

# TODO Give user option to make their own model


class Models:
    def __init__(self, hyperparameters, output_activation, name="model"):
        """
        Class to hold various NN architectures allowing for cleaner code when determining which architecture
        is best suited for DD.

        :param hyperparameters: a dictionary holding the parameters for the models:
        ('bin_array', 'dropout_rate', 'learning_rate', 'num_units')
        """
        self.hyperparameters = hyperparameters
        self.output_activation = output_activation
        self.name = name

    def original(self, input_shape):
        x_input = Input(input_shape, name="original")
        x = x_input
        for j, i in enumerate(self.hyperparameters['bin_array']):
            if i == 0:
                x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_%i" % (j + 1))(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            else:
                x = Dropout(self.hyperparameters['dropout_rate'])(x)
        x = Dense(1, activation=self.output_activation, name="Output_Layer")(x)
        model = Model(inputs=x_input, outputs=x, name='Progressive_Docking')
        return model

    def dense_dropout(self, input_shape):
        """This is the most simple neural architecture.
        Four dense layers, batch normalization, relu activation, and dropout.
        """
        # The model input
        x_input = Input(input_shape, name='dense_dropout')
        x = x_input

        # Model happens here...
        x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_1")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_2")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_3")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer_4")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        # output
        x = Dense(1, activation=self.output_activation, name="Output_Layer")(x)
        model = Model(inputs=x_input, outputs=x, name='Progressive_Docking')
        return model

    def wide_net(self, input_shape):
        """
        A simple square model
        """
        # The model input
        x_input = Input(input_shape, name='wide_net')
        x = x_input

        # The width coefficient
        width_coef = len(self.hyperparameters['bin_array'])//2

        for i, layer in enumerate(self.hyperparameters['bin_array']):
            if layer == 0:
                layer_name = "Hidden_Dense_Layer_" + str(i//2)
                x = Dense(self.hyperparameters['num_units'] * width_coef, name=layer_name)(x)
                x = Activation('relu')(x)
            else:
                x = BatchNormalization()(x)
                x = Dropout(self.hyperparameters['dropout_rate'])(x)

        # output
        x = Dense(1, activation=self.output_activation, name="Output_Layer")(x)
        model = Model(inputs=x_input, outputs=x, name='Progressive_Docking')
        return model

    def shared_layer(self, input_shape):
        """This Model uses a shared layer"""
        # The model input
        x_input = Input(input_shape, name="shared_layer")

        # Here is a layer that will be shared
        shared_layer = Dense(input_shape[0], name="Shared_Hidden_Layer")

        # Apply the layer twice
        x = shared_layer(x_input)
        x = shared_layer(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        # Apply dropout and normalization
        x = Dense(self.hyperparameters['num_units'], name="Hidden_Layer")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)

        for i, layer in enumerate(self.hyperparameters['bin_array']):
            if layer == 0:
                layer_name = "Hidden_Layer_" + str(i)
                x = Dense(self.hyperparameters['num_units']//i, name=layer_name)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(self.hyperparameters['dropout_rate'])(x)

        # output
        x = Dense(1, activation=self.output_activation, name="output_layer")(x)
        model = Model(inputs=x_input, outputs=x, name='Progressive_Docking')
        return model

    @staticmethod
    def get_custom_objects():
        return {"Lasso": Lasso}

    @staticmethod
    def get_available_modes():
        modes = []
        for attr in Models.__dict__:
            if attr[0] != '_' and attr != "get_custom_objects" and attr != "get_available_modes":
                modes.append(attr)
        return modes


class TunerModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_tuner_model(self, hp):
        """
        This method should be used with keras tuner.
        """

        # Create the hyperparameters
        num_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=4, step=1)
        num_units = hp.Int("num_units", min_value=128, max_value=1024)
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.8)
        learning_rate = hp.Float('learning_rate', min_value=0.00001, max_value=0.001)
        epsilon = hp.Float('epsilon', min_value=1e-07, max_value=1e-05)
        kernel_reg_func = [None, Lasso, l1, l2][hp.Choice("kernel_reg", values=[0, 1, 2, 3])]
        reg_amount = hp.Float("reg_amount", min_value=0.0, max_value=0.001, step=0.0001)

        # Determine how the layer(s) are shared
        share_layer = hp.Boolean("shared_layer")
        if share_layer:
            share_all = hp.Boolean("share_all")
            shared_layer_units = hp.Int("num_units", min_value=128, max_value=1024)
            shared_layer = Dense(shared_layer_units, name="shared_hidden_layer")
            if not share_all:
                where_to_share = set()
                layer_connections = hp.Int("num_shared_layer_connections", min_value=1, max_value=num_hidden_layers)
                for layer in range(layer_connections):
                    where_to_share.add(hp.Int("where_to_share", min_value=0, max_value=num_hidden_layers, step=1))

        # Build the model according to the hyperparameters
        inputs = Input(shape=self.input_shape, name="input")
        x = inputs

        # Determine number of hidden layers
        for layer_num in range(num_hidden_layers):
            # If we are not using a kernel regulation function or not...
            if kernel_reg_func is None:
                x = Dense(num_units, name="dense_" + str(layer_num))(x)
            else:
                x = Dense(num_units, kernel_regularizer=kernel_reg_func(reg_amount), name="dense_" + str(layer_num))(x)

            # If we are using a common shared layer, then connect it.
            if (share_layer and share_all) or (share_layer and layer_num in where_to_share):
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Dropout(dropout_rate)(x)
                x = shared_layer(x)

            # Apply these to every layer
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout_rate)(x)

        outputs = Dense(1, activation='sigmoid', name="output_layer")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', "AUC", "Precision", "Recall", DDMetrics.scaled_performance])

        print(model.summary())
        return model


