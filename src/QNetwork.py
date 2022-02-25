#!/usr/bin/env python3

import tensorflow as tf
import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Flatten, Dense


class QNetwork(Sequential):
    """
    Simple Q network with 4 layers, input, flatten, dense with relu and dense with linear activations
    """
    def __init__(self, input_size: int, n_actions: int = 2, hidden_size: int = 128):
        """
        Init
        :param input_size: number of features in input
        :param n_actions: number of actions in action pool
        :param hidden_size: hidden layer size
        """
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.input_layer = Input(shape=(1, self.input_size))
        self.flatten_layer = Flatten(input_shape=(1, self.input_size))(self.input_layer)
        self.dense_layer = Dense(self.hidden_size,
                            activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01))(self.flatten_layer)
        self.output_layer = Dense(n_actions, activation='linear')(self.dense_layer)

        self.add(self.input_layer)
        self.add(self.flatten_layer)
        self.add(self.dense_layer)
        self.add(self.output_layer)
