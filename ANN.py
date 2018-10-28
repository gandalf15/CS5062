#!/usr/bin/env python3
import numpy as np
class FeedForwardANN:
    """
    This class represents a FeedForward ANN with N layers.
    It is a vectorised implementation with the help of numpy.
    Theta = weights
    delta = error
    """
    def __init__(self, inputs, input_type, layers, outputs, err_func, act_func):
        """
        Init the FF-ANN

        Args:
            inputs(int): Number of inputs for the ANN
            layers(int): Number of hidden layers. Excluding input layer and output layer.
            outputs(int): Number of outputs for the ANN
            err_func(function): Error function.
            act_func(function): Activation function for neurons of the ANN
        """
        self._inputs = inputs
        self._input_type = input_type
        self._layers = layers
        self._outputs = outputs
        self._err_func = err_func
        self._act_func = act_func

        # Construct the architecture of the ANN
        self._current_input = np.zeros(self._inputs, dtype=)
        # Init parameters (weights) and add +1 for bias unit (neuron)
        self.thetas = np.random.rand()

    @property
    def inputs(self):
        """inputs getter"""
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """inputs setter"""
        if not isinstance(value, int):
            raise TypeError("Number of inputs of ANN must be an integer!")
        if value < 1:
            raise ValueError("Number of inputs of ANN must be a positive integer!")
        self._inputs = value
    
    @property
    def input_type(self):
        """input_type getter"""
        return self._input_type

    @input_type.setter
    def input_type(self, value):
        """input_type setter"""
        if not type(value).__module__ == np.__name__:
            raise TypeError("input_type must be a numpy data type")
        self._input_type = value

    @property
    def layers(self):
        """layers getter"""
        return self._layers

    @layers.setter
    def layers(self, value):
        """layers setter"""
        if not isinstance(value, int):
            raise TypeError("Number of layers of ANN must be an integer!")
        if value < 0:
            raise ValueError("Number of layers of ANN must be a zero or more!")
        self._layers = value

    @property
    def outputs(self):
        """outputs getter"""
        return self._outputs
    
    @outputs.setter
    def outputs(self, value):
        """outputs setter"""
        if not isinstance(value, int):
            return TypeError("Number of outputs of ANN must be an integer!")
        if value < 1:
            return ValueError("Number of outputs of ANN must be a positive integer!")
        self._outputs = value

    @property
    def err_func(self):
        """err_func getter"""
        return self._err_func

    @property
    def act_func(self):
        """act_func getter"""
        return self._act_func