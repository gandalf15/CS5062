#!/usr/bin/env python3
import numpy as np
from ann import FeedForwardANN
from act_functions import Threshold
from act_functions import Sigmoid

a = FeedForwardANN(2, 1, 2, 1, Sigmoid())
a._thetas = [[np.array([0.0, 0.1, 0.8]),
              np.array([0.0, 0.4, 0.6])], [np.array([0.0, 0.3, 0.9])]]
print("Printing Neurons:")
print(a._neurons)
print("\nPrinting Thetas:")
print(a._thetas)
print("\nPrinting calculated_neuron_inputs")
print(a._calculated_neuron_inputs)
print("\nFF:")
a.feed_forward(np.array([0.35, 0.9]))
print("\n")
print("Printing Neurons:")
print(a._neurons)
print("\nPrinting Thetas:")
print(a._thetas)
print("\nPrinting calculated_neuron_inputs")
print(a._calculated_neuron_inputs)