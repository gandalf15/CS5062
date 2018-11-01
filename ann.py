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
            self._thetas.append([
                2 * np.random.rand(self._inputs + 1) - 1
                for j in range(self._h_neurons)
            ])
            self._calculated_neuron_inputs.append(
                np.zeros((self._h_neurons, self._inputs + 1)))
            self._neurons = self._neurons + [
                np.ones(self._h_neurons + 1) for j in range(self._h_layers)
            ]
            for i in range(self._h_layers - 1):
                self._thetas.append([
                    2 * np.random.rand(self._h_neurons + 1) - 1
                    for j in range(self._h_neurons)
                ])
                self._calculated_neuron_inputs.append(
                    np.zeros((self._h_neurons, self._h_neurons + 1)))
        self._neurons.append(np.ones(self._outputs))
        self._thetas.append([
            2 * np.random.rand(self._h_neurons + 1) - 1
            for j in range(self._outputs)
        ])
        self._calculated_neuron_inputs.append(
            np.zeros((self._outputs, self._h_neurons + 1)))

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
        if len(input_arr) != self._inputs:
            raise ValueError(
                "input_arr is not the same size as number of ANN inputs!")

        # update input and do not touch bias unit
        for i in range(self._inputs):
            self._neurons[0][i + 1] = input_arr[i]

        iters = range(len(self._thetas))
        for i in iters:
            # print("self._thetas[{}]".format(i))
            # print(self._thetas[i])
            # print("self._neurons[{}]".format(i))
            # print(self._neurons[i])
            # print("self._thetas[{}] * self._neurons[{}])".format(i,i))
            # print(self._thetas[i] * self._neurons[i])

            # print("np.array([np.sum(row) for row in (self._thetas[{}] * self._neurons[{}])])".format(i,i))
            # print(np.array([np.sum(row) for row in (self._thetas[i] * self._neurons[i])]))
            # print()
            self._calculated_neuron_inputs[
                i] = self._thetas[i] * self._neurons[i]
            neurons_inputs_sum = np.array(
                [np.sum(row) for row in self._calculated_neuron_inputs[i]])
            if i == iters[-1]:
                self._neurons[i + 1] = self._act_func(neurons_inputs_sum)
            else:
                self._neurons[i + 1][1:] = self._act_func(neurons_inputs_sum)

        return self._neurons[-1]

    def back_propagation(self, expected_out, learning_rate=0.1):
        """
        back propagation for ANN
        Args:
            expected_out(np.array): array of expected output from the ANN
            learning_rate(float): learning rate
        Raises:
            ValueError: if the len(expected out) != size of output of the ANN.
        """
        if len(expected_out) != self._outputs:
            raise ValueError(
                "expected_out is not the same size as number of outputs of the ANN!"
            )
        deltas = []
        # calculate error for the output layer
        deltas.append(expected_out - self._neurons[-1])

        iters = range(len(self._thetas))
        for i in iters[:0:-1]:
            # print("deltas")
            # print(deltas)
            # print("self._thetas[{}] * deltas[-1]".format(i))
            # print((self._thetas[i] * deltas[-1]))
            # print("self._neurons[{}]".format(i))
            # print(self._neurons[i])
            # print("(1 - self._neurons[{}])".format(i))
            # print((1 - self._neurons[i]))
            if self._act_func.__name__ == 'sigmoid':
                deltas.append(self._calculated_neuron_inputs[i] *
                              (1 - self._calculated_neuron_inputs[i]) *
                              (np.sum(self._thetas[i] * deltas[-1], axis=0)))
            else:
                deltas.append(np.sum(self._thetas[i] * deltas[-1], axis=0))
        i = 0
        for j in range(len(self._thetas))[::-1]:
            print("old thetas")
            print(self._thetas[j])
            print("self._neurons[j]")
            print(self._neurons[j])
            print("deltas[i]")
            print(deltas[i])
            self._thetas[j] -= learning_rate *  deltas[i] * self._neurons[j]
            print("new thetas")
            print(self._thetas[j])
            i += 1

    def train(self, training_set_input, expected_output, iterations, learning_rate=0.1):
        """
        train method starts training of weights of the ANN
        Args:
            training_set(np.array): 2d array with training examples. Last element is expected output
            expected_output(np.array): 2d array with expected output for training set
            iterations(int): max number of iterations
            learning_rate(float): learning rate
        Raises:
            ValueError: if the dimensions of training set are not correct
        Returns(float):Error rate at the end of training
        """
        if training_set_input.shape[1] != self._inputs:
            raise ValueError(
                "training_set does not have rght amount of columns.\nExpected ",
                self._inputs, " Instead I got: ", training_set_input.shape[1])
        err = 0.0
        for i in range(iterations):
            for j in range(len(training_set_input)):
                hypothesis = self.feed_forward(training_set_input[j])
                err = np.mean(np.abs(hypothesis - expected_output[j]))
                self.back_propagation(expected_output[j], learning_rate)
            if i % 1000 == 0:
                print("Iteration: ", i, "Mean Error: ", err)
            err = 0.0
        return err