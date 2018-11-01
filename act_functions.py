#!/usr/bin/env python3
import numpy as np

def sigmoid(values):
    """
    sigmoid function

    Args:
        values(np.array): Input array of float64 to sigmoid function
    Raises:
        ValueError: If provided list does not contain only floats
    Returns: np.float64
    """
    return 1 / (1 + np.exp(-values))

def sig_to_deriv(values):
    return values*(1-values)

def threshold(values):
    """
    threshold function
    Args:
        values(np.array): Input array of np.float64 to sign function
    Returns(float): result
    """
    sum_of_values = np.sum(values)
    result = [-1.0]
    if sum_of_values >= 0.0:
        result = [1.0]
    return np.array(result)