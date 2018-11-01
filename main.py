#!/usr/bin/env python3
import numpy as np
from ann2 import FeedForwardANN
from act_functions import threshold
from act_functions import sigmoid

nn = FeedForwardANN(3, 12, 1, sigmoid)

training_set = np.array([[0,0,1],
                        [0,1,1],
                        [1,0,1],
                        [1,1,1]])
expected_out = np.array([[0],
                        [1],
                        [1],
                        [0]])
learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
for rate in learning_rates:
    print("Learning rate: {}".format(rate))
    nn.train(training_set, expected_out, 40000, learning_rate=rate)
    print()
