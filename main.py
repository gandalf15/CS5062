#!/usr/bin/env python3
import numpy as np
from ann import FeedForwardANN
from act_functions import threshold
from act_functions import sigmoid

with open("../phishing-dataset/training_dataset.arff") as f:
    data = np.loadtxt(f, dtype=float, delimiter=',', ndmin=2)

training_set = []
expected_out = []

for row in data:
    training_set.append(row[:-1])
    expected_out.append(row[-1])

training_set = np.array(training_set)
expected_out = np.array(expected_out)

print(training_set.shape)
print(expected_out.shape)

nn = FeedForwardANN(
    inputs=len(training_set[0]), h_neurons=32, outputs=1, act_func=sigmoid)

learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
for rate in learning_rates:
    print("Learning rate: {}".format(rate))
    nn.train(training_set, expected_out, 40000, learning_rate=rate)
    print()
