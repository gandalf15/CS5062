#!/usr/bin/env python3
import numpy as np


class FeedForwardANN:
    """
    This class represents a FeedForward ANN with N layers.
    It is a vectorised implementation with the help of numpy.
    Theta = weights
    delta = error
    """

    def __init__(self, inputs, input_type, h_layers, h_neurons, outputs,
                 err_func, act_func):
        """
        Init the FF-ANN

        Args:
            inputs(int): Number of inputs for the ANN
            input_type(numpy type): Data type of the input
            h_layers(int): Number of hidden layers. Excluding input layer and output layer.
            h_neurons(int): Number of neurons in each hidden layer
            outputs(int): Number of outputs for the ANN
            err_func(function): Error function.
            act_func(function): Activation function for neurons of the ANN
        """
        self._inputs = inputs
        self._input_type = input_type
        self._h_layers = h_layers
        self._h_neurons = h_neurons
        self._outputs = outputs
        self._err_func = err_func
        self._act_func = act_func

        # Construct the architecture of the ANN
        # Init the neurons and do not forget to create +1 bias unit
        self._neurons = []
        # Init parameters (weights) with random values between 0 to 1
        # and add +1 for bias unit (neuron).
        # Each neuron has 1..N params and each layer has 1..N neurons.
        # Therefore, we need 3D array
        self._thetas = []
        # the first layer is not neuron layer but an input values
        self._neurons.append(np.ones(self._inputs + 1, dtype=self._input_type))
        # Check what is the act_funct and then set the data type for neuron values
        if self._act_func.__name__ == "sigmoid":
            dtype = np.float64
        elif self._act_func.__name__ == "sign":
            dtype = np.bool_  # default for sgn function
        else:
            raise TypeError(
                "Unknown act_func. Need to know for data type. Edit __init__ method"
            )

        for i in range(self._h_layers):
            self._neurons.append(np.ones(self._h_neurons + 1, dtype=dtype))
            if i == 0:
                self._thetas.append(
                    np.random.rand((self._h_neurons + 1) * self._inputs))
            elif i == self._h_layers - 1:
                self._thetas.append(
                    np.random.rand((self._h_neurons + 1) * self._outputs))
            else:
                self._thetas.append(
                    np.random.rand((self._h_neurons + 1) * self._h_neurons))

        self._neurons.append(np.ones(self._outputs, dtype=dtype))

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
        return self._h_layers

    @layers.setter
    def layers(self, value):
        """layers setter"""
        if not isinstance(value, int):
            raise TypeError("Number of layers of ANN must be an integer!")
        if value < 0:
            raise ValueError("Number of layers of ANN must be a zero or more!")
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
    def err_func(self):
        """err_func getter"""
        return self._err_func

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


class Sigmoid:

    @staticmethod
    def sigmoid(values):
        """
        sigmoid function

        Args:
            values(list): Input list of floats to sigmoid function
        Raises:
            ValueError: If provided list does not contain only floats
        Returns: float
        """
        #TODO
        pass

    @staticmethod
    def err_func(expected_out, actual_output):
        """
        err_func for the sigmoid function
        Args:
            expected_out(float): expected output from the neuron
            actual_out(float): actual output from the neuron
        Raises:
            ValueError: If expected_out or actual_out is not float type
        Returns: calculated error
        """
        #TODO
        pass


class Sign:

    @staticmethod
    def sign(values):
        """
        sign function
        Args:
            values(list): Input list of floats to sign function
        Raises:
            ValueError: If provided list does not contain only floats
        Returns: float
        """
        #TODO
        pass

    @staticmethod
    def err_func(expected_out, actual_output):
        """
        err_func for sign function
        Args:
            expected_out(int): expected output from the neuron
            actual_out(int): actual output from the neuron
        Raises:
            ValueError: If expected_out or actual_out is not int type
        Returns: calculated error
        """
        #TODO
        pass
