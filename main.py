#!/usr/bin/env python3
import numpy as np

from act_functions import sigmoid
from ann import FeedForwardANN
from sklearn.model_selection import KFold

training_set = []
expected_out = []
with open("../phishing-dataset/training_dataset_updated") as f:
    data = np.loadtxt(f, dtype=float, delimiter=',', ndmin=2)

for row in data:
    training_set.append(row[1:])
    expected_out.append([row[0]])

training_set = np.array(training_set)
expected_out = np.array(expected_out)

# print(expected_out[:2])
# print(training_set[:2])

# print(training_set.shape)
# print(expected_out.shape)

# nn.load_model(dir_path='saved_models', file_name='2018_11_02~17-15-52_iter_10000')

# learning_rates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# learning_rates = reversed(learning_rates)
# for rate in learning_rates:
# print("Learning rate: {}".format(rate))

neurons = [14]
iterations = 5000
learning_rate = 0.003
for neuron_count in neurons:
    errors = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(training_set):
        # print("Training set length: ", len(training_set[train_index]))
        # print("Training expected set length: ", len(expected_out[train_index]))
        # print("Test set length: ", len(training_set[test_index]))
        # print("Test expected set length: ", len(expected_out[test_index]))
        nn = FeedForwardANN(
        inputs=len(training_set[0]), h_neurons=neuron_count, h_layers=1, outputs=1, act_func=sigmoid)
        nn.train(
            training_set=training_set[train_index],
            expected_output=expected_out[train_index],
            iterations=iterations,
            checkpoint=1000,
            learning_rate=learning_rate,
            batch_size=0)
        hypothesis = nn.feed_forward(training_set[test_index])
        errors.append(np.mean((hypothesis - expected_out[test_index])**2))
        print("Current MSError: ", errors[-1])
    confusion_matrix = [[0,0], [0,0]]
    for i in range(len(expected_out)):
        result = nn.feed_forward(np.array([training_set[i]]))
        if int(result[0]+0.5) == int(expected_out[i][0]) and int(result[0]+0.5) == 1:
            confusion_matrix[0][0] += 1
        elif int(result[0]+0.5) == int(expected_out[i][0]) and int(result[0]+0.5) == 0:
            confusion_matrix[1][1] += 1
        elif int(result[0]+0.5) != int(expected_out[i][0]) and int(result[0]+0.5) == 1:
            confusion_matrix[1][0] += 1
        elif int(result[0]+0.5) != int(expected_out[i][0]) and int(result[0]+0.5) == 0:
            confusion_matrix[0][1] += 1
    print("confusion matrix: \n", ' '*6,"Positive predict | Negative predict")
    print('-'*40)
    print('Positive actual | ', confusion_matrix[0][0], ' | ', confusion_matrix[0][1])
    print('Negative actual | ', confusion_matrix[1][0], '  | ', confusion_matrix[1][1])
    print()
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1]) / len(training_set)
    print("Accuracy = ", acc*100, '%')
    acc_pos = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    print('Accuracy Positive / Recall: ', acc_pos*100, '%')
    acc_neg = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    print('Accuracy Negative: ', acc_neg*100, '%')
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    print('Prediction: ', precision*100, '%')
    f_measure = 2*(acc_pos * precision) / (acc_pos + precision)
    print('F-measure: ', f_measure)
    print()
    print()
    print("-"*80)
    print("Number of HL neurons: ", neuron_count)
    print("Learning rate: ", learning_rate)
    print("10-fold CV MSError: ", np.mean(errors))

# for i in range(len(expected_out[:10])):
#     print("Input: ", training_set[i])
#     print("prediction: ", nn.feed_forward(np.array([training_set[i]])))
#     print("expected_out ", expected_out[i])
