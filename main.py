#!/usr/bin/env python3
from time import gmtime, strftime

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold

from act_functions import sigmoid
from ann import FeedForwardANN


def show_roc_curve(hypothesis, actual_class, legenda):
    plt.cla()
    plt.title('Receiver Operating Characteristic')
    for i in range(len(legenda)):
        fpr, tpr, thresholds = metrics.roc_curve(actual_class[i], hypothesis[i])
        roc_auc = metrics.auc(fpr, tpr)
        label = legenda[i] + ', {0:.3f} AUC'.format(roc_auc)
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], 'r--', label='random')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='best')
    plt.show()


def save_roc_curve(hypothesis, actual_class, legenda):
    plt.cla()
    plt.title('Receiver Operating Characteristic')
    for i in range(len(legenda)):
        fpr, tpr, thresholds = metrics.roc_curve(actual_class[i], hypothesis[i])
        roc_auc = metrics.auc(fpr, tpr)
        label = legenda[i] + ', {0:.3f}'.format(roc_auc)
        # label = 'perceptron' + ', {0:.3f} AUC'.format(roc_auc)
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], 'r--', label='random')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='best')
    file_name = strftime("%Y_%m_%d~%H-%M-%S", gmtime()) + '.png'
    plt.savefig('ROC~' + file_name)


def make_confusion_matrix(hypothesis, actual_class):
    confusion_matrix = [[0, 0], [0, 0]]
    for i in range(len(hypothesis)):
        if int(hypothesis[i][0] + 0.5) == int(
                actual_class[i][0]) and int(hypothesis[i][0] + 0.5) == 1:
            confusion_matrix[0][0] += 1
        elif int(hypothesis[i][0] + 0.5) == int(
                actual_class[i][0]) and int(hypothesis[i][0] + 0.5) == 0:
            confusion_matrix[1][1] += 1
        elif int(hypothesis[i][0] + 0.5) != int(
                actual_class[i][0]) and int(hypothesis[i][0] + 0.5) == 1:
            confusion_matrix[1][0] += 1
        elif int(hypothesis[i][0] + 0.5) != int(
                actual_class[i][0]) and int(hypothesis[i][0] + 0.5) == 0:
            confusion_matrix[0][1] += 1
    return np.array(confusion_matrix)


training_set = []
expected_out = []
with open("../phishing-dataset/training_dataset_updated") as f:
    data = np.loadtxt(f, dtype=float, delimiter=',', ndmin=2)

for row in data:
    training_set.append(row[1:])
    expected_out.append([row[0]])

training_set = np.array(training_set)
expected_out = np.array(expected_out)

# nn.load_model(dir_path='saved_models', file_name='2018_11_02~17-15-52_iter_10000')

h_neurons = [20]
h_layers = 1
iterations = 1000
learning_rate = 0.01
checkpoint = 0
all_predictions = []
all_classes = []
all_labels = []
for neuron_count in h_neurons:
    errors = []
    kf = KFold(n_splits=10)
    confusion_matrix = np.array([[0, 0], [0, 0]])
    model_predictions = []
    model_classes = []
    for train_index, test_index in kf.split(training_set):
        nn = FeedForwardANN(
            inputs=len(training_set[0]),
            h_neurons=neuron_count,
            h_layers=h_layers,
            outputs=1,
            act_func=sigmoid)
        nn.train(
            training_set=training_set[train_index],
            expected_output=expected_out[train_index],
            iterations=iterations,
            checkpoint=checkpoint,
            learning_rate=learning_rate,
            batch_size=0)
        hypothesis = nn.feed_forward(training_set[test_index])
        actual_class = expected_out[test_index]
        errors.append(np.mean((hypothesis - actual_class)**2))
        print("Current MSError: ", errors[-1])
        confusion_matrix += make_confusion_matrix(hypothesis, actual_class)
        model_predictions.extend(hypothesis.flatten())
        model_classes.extend(actual_class.flatten())
    all_predictions.append(np.array(model_predictions))
    all_classes.append(np.array(model_classes))
    all_labels.append("{} H_L, {} H_N".format(h_layers, neuron_count))
    print("-" * 80)
    print("Number of HL neurons: ", neuron_count)
    print("Number of H layers: ", h_layers)
    print("Learning rate: ", learning_rate)
    print("Iterations: ", iterations)
    print("10-fold CV MSError: ", np.mean(errors))

    print("confusion matrix: \n", ' ' * 6,
          "Positive predict | Negative predict")
    print('-' * 40)
    print('Positive actual | ', confusion_matrix[0][0], ' | ',
          confusion_matrix[0][1])
    print('Negative actual | ', confusion_matrix[1][0], '  | ',
          confusion_matrix[1][1])
    print()
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1]) / len(training_set)
    print("Accuracy = ", acc)
    acc_pos = confusion_matrix[0][0] / (
        confusion_matrix[0][0] + confusion_matrix[0][1])
    print('Accuracy Positive / Recall: ', acc_pos)
    acc_neg = confusion_matrix[1][1] / (
        confusion_matrix[1][1] + confusion_matrix[1][0])
    print('Accuracy Negative: ', acc_neg)
    precision = confusion_matrix[0][0] / (
        confusion_matrix[0][0] + confusion_matrix[1][0])
    print('Prediction: ', precision)
    f_measure = 2 * (acc_pos * precision) / (acc_pos + precision)
    print('F-measure: ', f_measure)
    print()
    print()
    print("-" * 80)

save_roc_curve(
    hypothesis=all_predictions, actual_class=all_classes, legenda=all_labels)
# show_roc_curve(hypothesis=all_predictions, actual_class=all_classes, legenda=all_labels)

# for i in range(len(expected_out[:10])):
#     print("Input: ", training_set[i])
#     print("prediction: ", nn.feed_forward(np.array([training_set[i]])))
#     print("expected_out ", expected_out[i])
