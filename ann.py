#!/usr/bin/env python3
import os
import pickle
from time import gmtime, strftime

import numpy as np

from act_functions import sig_to_deriv


class FeedForwardANN:
    """
    This class represents a FeedForward ANN with N layers.
    It is a vectorised implementation with the help of numpy.
    I sometime refer to:
    Theta = weights
    delta = error
    """

    def __init__(self, inputs, h_neurons, h_layers, outputs, act_func):
        """
        Init the FF-ANN

        Args:
            inputs(int): Number of inputs for the ANN
            h_neurons(int): Number of neurons in each hidden layer
            outputs(int): Number of outputs for the ANN
            act_func(function): Activation function for neurons of the ANN
        """
        #TODO: multiple layers
        np.random.seed(1)
        self._inputs = inputs
        self._h_neurons = h_neurons
        self._h_layers = h_layers
        self._outputs = outputs
        self._act_func = act_func
        # Construct the architecture of the ANN
        self._neurons = []
        self._thetas = []
        # the first layer is an input layer
        self._thetas.append((2 * np.random.random(
            (self._inputs, self._h_neurons)) - 1) * 0.01)
        for i in range(self._h_layers - 1):
            self._thetas.append((2 * np.random.random(
                (self._h_neurons, self._h_neurons)) - 1) * 0.01)
        self._thetas.append((2 * np.random.random(
            (self._h_neurons, self._outputs)) - 1) * 0.01)

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
        Returns(np.array): array of output values from the ANN
        """
        self._neurons.append(input_arr)
        for i in range(len(self._thetas)):
            self._neurons.append(
                self._act_func(np.dot(self._neurons[-1], self._thetas[i])))
        return self._neurons[-1]

    def back_propagation(self, expected_out, learning_rate=0.001):
        """
        back propagation for ANN
        Args:
            expected_out(np.array): array of expected output from the ANN
            learning_rate(float): learning rate
        Raises:
            ValueError: if the len(expected out) != size of output of the ANN.
        """
        errors, deltas = [], []
        errors.append(self._neurons[-1] - expected_out)
        if self._act_func.__name__ == 'sigmoid':
            for i in reversed(range(1,len(self._neurons))):
                deltas.append(sig_to_deriv(self._neurons[i]) * errors[-1])
                if i != 1:
                    errors.append(deltas[-1].dot((self._thetas[i-1]).T))
        else:
            raise TypeError('Unknown activation function')
        j = len(deltas)-1
        for i in range(len(deltas)):
            self._thetas[i] -= learning_rate * (self._neurons[i]).T.dot(deltas[j])
            j -= 1
        self._neurons = []

    def save_model(self, dir_path, file_name):
        """
        It saves the current model of the NN
        
        Args:
            dir_path(str): Path to dir where to store the models
            file_name(str): file name of the model
        """
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        full_path = os.path.join(dir_path, file_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self._thetas, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, dir_path, file_name):
        """
        It loads the specified model from the path provided

        Args:
            dir_path(str): Path to dir where the models are
            file_name(str): file name of the model
        """
        full_path = os.path.join(dir_path, file_name)
        with open(full_path, 'rb') as f:
            self._thetas = pickle.load(f)

    def train(self,
              training_set,
              expected_output,
              iterations,
              batch_size=0,
              learning_rate=0.001,
              checkpoint=0):
        """
        train method starts training of weights of the ANN
        Args:
            training_set(np.array): 2d array with training examples.
            expected_output(np.array): 2d array with expected output for training set
            batch_size(int): How many examples to use at once. Default 0 means all
            iterations(int): max number of iterations
            learning_rate(float): learning rate
            checkpoint(int): Save the model every N epoch. Default 0 means do not save.
        Raises:
            ValueError: if the dimensions of training set are not correct
        Returns(float):Error rate at the end of training
        """
        #TODO: Implement batch size training
        err = 0.0
        for i in range(iterations):
            hypothesis = self.feed_forward(training_set)
            self.back_propagation(expected_output, learning_rate)
            err = np.mean((hypothesis - expected_output)**2)
            if i % 500 == 0:
                print("Iteration: ", i, "Mean Squared Error: ", err)
            if checkpoint and i % checkpoint - 1 == 0 and i != 0:
                file_name = strftime("%Y_%m_%d~%H-%M-%S_iter_",
                                     gmtime()) + str(i)
                self.save_model(dir_path='./saved_models/', file_name=file_name)

        return err


# err = 0.0
#         if batch_size == 0:
#             batch_size = len(training_set)
#         num_of_slices = int(len(training_set) / batch_size)
#         batched_training_set = np.array_split(training_set, num_of_slices)
#         batched_expected_output = np.array_split(expected_output, num_of_slices)
#         for i in range(iterations):
#             for j in range(num_of_slices):
#                 hypothesis = self.feed_forward(batched_training_set[j])
#                 self.back_propagation(batched_expected_output[j], learning_rate)
#             err = np.mean(np.abs(hypothesis - expected_output[j]))
#             if i % 100 == 0:
#                 print("Iteration: ", i, "Mean Error: ", err)
#             if checkpoint and i % checkpoint == 0 and i != 0:
#                 file_name = strftime("%Y_%m_%d~%H-%M-%S_iter_", gmtime()) + str(i)
#                 self.save_model(dir_path='./saved_models/', file_name=file_name)

#         return err