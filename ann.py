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
        self._calculated_neuron_inputs = []
        # the first layer is an input layer

        self._neurons.append(np.ones(self._inputs + 1))
        if self._h_layers > 0:
            self._thetas.append([np.random.rand(self._inputs + 1) for j in range(self._h_neurons)])
            self._calculated_neuron_inputs.append(np.zeros((self._h_neurons, self._inputs + 1)))
            self._neurons = self._neurons + [np.ones(self._h_neurons + 1) for j in range(self._h_layers)]
            #self._neurons.append(np.ones((self._h_layers, self._h_neurons + 1)))
            for i in range(self._h_layers - 1):
                self._thetas.append([np.random.rand(self._h_neurons + 1) for j in range(self._h_neurons)])
                self._calculated_neuron_inputs.append(np.zeros((self._h_neurons, self._h_neurons + 1)))
        self._neurons.append(np.ones(self._outputs))
        self._thetas.append([np.random.rand(self._h_neurons + 1) for j in range(self._outputs)])
        self._calculated_neuron_inputs.append(np.zeros((self._outputs, self._h_neurons + 1)))
    
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

    def feed_forward(self, input_arr):
        """
        feed forward method for ANN
        Args:
            input_arr(np.array): array of input values. It must be the same size as inputs.
        Raises:
            ValueError: if the size of array is not the same as number of inputs fot the ANN
        Returns(np.array): array of output values from the ANN
        """
        if len(input_arr) != self.inputs:
            raise ValueError(
                "input_arr is not the same size as number of ANN inputs!")
        # update input and do not touch bias unit
        for i in range(self._inputs):
            self._neurons[0][i+1] = input_arr[i]
        
        iters = range(len(self._calculated_neuron_inputs))
        for i in iters:
            self._calculated_neuron_inputs[i] = self._thetas[i] * self._neurons[i]
            temp = []
            for neuron_inputs in self._calculated_neuron_inputs[i]:
                temp.append(self._act_func.act_func(neuron_inputs))
            if i != iters[-1]:
                self._neurons[i+1][1:] = np.array(temp)
            else:
                self._neurons[i+1] = np.array(temp)
            # print(i)
            # print(self._calculated_neuron_inputs[i])
        
        
    def back_propagation(self):
        """back propagation for ANN"""
        #TODO
        pass
