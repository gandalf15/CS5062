#!/usr/bin/env python3
import numpy as np


class Sigmoid:

    def act_func(self, values):
        """
        sigmoid function

        Args:
            values(np.array): Input array of float64 to sigmoid function
        Raises:
            ValueError: If provided list does not contain only floats
        Returns: np.float64
        """
        sum_of_values = np.sum(values)
        return 1 / (1 + np.exp(-sum_of_values))

    def err_func(self, expected_out, actual_output):
        """
        err_func for the sigmoid function
        Args:
            expected_out(float): expected output from the neuron
            actual_out(float): actual output from the neuron
        Returns(float): calculated error
        """
        return actual_output * (1 - actual_output) * (
            expected_out - actual_output)


class Sign:

    def act_func(self, values):
        """
        sign function
        Args:
            values(np.array): Input array of np.float64 to sign function
        Returns: int
        """
        sum_of_values = np.sum(values)
        result = 0.0
        if sum_of_values < 0.0:
            result = -1.0
        elif sum_of_values == 0.0:
            result = 0.0
        elif sum_of_values > 0.0:
            result = 1.0
        return result

    def err_func(self, expected_out, actual_output):
        """
        err_func for sign function
        Args:
            expected_out(float): expected output from the neuron
            actual_out(float): actual output from the neuron
        Returns(float): calculated error
        """
        return expected_out - actual_output


class Threshold:

    def act_func(self, values):
        """
        threshold function
        Args:
            values(np.array): Input array of np.float64 to sign function
        Returns(float): result
        """
        sum_of_values = np.sum(values)
        result = 0.0
        if sum_of_values >= 0.5:
            result = 1.0
        return result

    def err_func(self, expected_out, actual_output):
        """
        threshold for sign function
        Args:
            expected_out(float): expected output from the neuron
            actual_out(float): actual output from the neuron
        Returns(float): calculated error int
        """
        return expected_out - actual_output