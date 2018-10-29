#!/usr/bin/env python3
import numpy as np


class FeedForwardANN:
    """
    This class represents a FeedForward ANN with N layers.
    It is a vectorised implementation with the help of numpy.
    I sometime refer to:
    Theta = weights
    delta = error
    """

    def __init__(self, inputs, h_layers, h_neurons, outputs, act_func):
        """
        Init the FF-ANN

        Args:
            inputs(int): Number of inputs for the ANN
            h_layers(int): Number of hidden layers. Excluding input layer and output layer.
            h_neurons(int): Number of neurons in each hidden layer
            outputs(int): Number of outputs for the ANN
            act_func(function): Activation function for neurons of the ANN
        """
        self._inputs = inputs
        self._h_layers = h_layers
        self._h_neurons = h_neurons
        self._outputs = outputs
        self._act_func = act_func

        # Construct the architecture of the ANN
        # Init the neurons and do not forget to create +1 bias unit
        self._neurons = []
        # Init parameters (weights) with random values between 0 to 1
        # and add +1 for bias unit (neuron).
        # Each neuron has 1..N params and each layer has 1..N neurons.
        # Therefore, we have 3D array (list of 2D arrays)
        self._thetas = []
        # the first layer is an input layer
        dtype = np.float_
        self._neurons.append(np.ones(self._inputs + 1, dtype=dtype))

        for i in range(self._h_layers):
            self._neurons.append(np.ones(self._h_neurons + 1, dtype=dtype))
            if i == 0:
                self._thetas.append(
                    np.random.rand((self._h_neurons + 1) * self._inputs))
            else:
                self._thetas.append(
                    np.random.rand((self._h_neurons + 1) * self._h_neurons + 1))

        self._neurons.append(np.ones(self._outputs, dtype=dtype))
        if self._h_layers > 0:
            self._thetas.append(
                np.random.rand((self._h_neurons + 1) * self._outputs))
        else:
            self._thetas.append(
                np.random.rand((self._inputs + 1) * self._outputs))

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
            raise ValueError(
                "Number of inputs of ANN must be a positive integer!")
        self._inputs = value

    @property
    def h_layers(self):
        """h_layers getter"""
        return self._h_layers

    @h_layers.setter
    def h_layers(self, value):
        """h_layers setter"""
        if not isinstance(value, int):
            raise TypeError("Number of h_layers of ANN must be an integer!")
        if value < 0:
            raise ValueError(
                "Number of h_layers of ANN must be a zero or more!")
        self._h_layers = value

    @property
    def h_neurons(self):
        """h_neurons getter"""
        return self._h_neurons

    @h_neurons.setter
    def h_neurons(self, value):
        """h_neurons setter"""
        if not isinstance(value, int):
            raise TypeError("Number of h_neurons of ANN must be an integer!")
        if value < 1:
            raise ValueError(
                "Number of h_neurons of ANN must be greater than zero!")
        self._h_neurons = value

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
            return ValueError(
                "Number of outputs of ANN must be a positive integer!")
        self._outputs = value

    @property
    def act_func(self):
        """act_func getter"""
        return self._act_func

    def feed_forward(self):
        """feed forward method for ANN"""
        #TODO
        pass

    def back_propagation(self):
        """back propagation for ANN"""
        #TODO
        pass
