#!/usr/bin/env python3
import numpy as np
from ann import FeedForwardANN
from act_functions import threshold
from act_functions import sigmoid

training_set = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
            [1,1,0]])
expected_out = np.array([[0],
			[1],
			[1],
			[0],
            [0]])

# training_set = []
# expected_out = []
# with open("../phishing-dataset/training_dataset_updated") as f:
#     data = np.loadtxt(f, dtype=float, delimiter=',', ndmin=2)

# for row in data:
#     training_set.append(row[1:])
#     expected_out.append([row[0]])

# print(expected_out[:2])
# print(training_set[:2])
# training_set = np.array(training_set)
# expected_out = np.array(expected_out)

# print(training_set.shape)
# print(expected_out.shape)

nn = FeedForwardANN(
    inputs=len(training_set[0]), h_neurons=10, outputs=1, act_func=sigmoid)

learning_rates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
learning_rates = reversed(learning_rates)
for rate in learning_rates:
    print("Learning rate: {}".format(rate))
    nn.train(training_set, expected_out, 1000, learning_rate=rate)
    print()

for i in range(len(expected_out)):
    print("Input: ", training_set[i])
    print("prediction: ", nn.feed_forward(np.array([training_set[i]])))
    print("expected_out ", expected_out[i])
# print("Input: ", np.array([training_set[0]]))
# print("Output: ", nn.feed_forward(np.array([training_set[0]])))
# print("expected: ", np.array([expected_out[0]]))

# print("Input: ", np.array([training_set[1]]))
# print("Output: ", nn.feed_forward(np.array([training_set[1]])))
# print("expected: ", np.array([expected_out[1]]))

# print("Input: ", np.array([training_set[4]]))
# print("Output: ", nn.feed_forward(np.array([training_set[4]])))
# print("expected: ", np.array([expected_out[4]]))