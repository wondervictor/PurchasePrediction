# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
import numpy as np


def fscore(truth, pred):
    pass


def evaluate(pred_prob, labels):
    """
    验证：精确率，召回率，F score
    :param pred_prob:
    :param labels:
    :return:
    """
    result = [np.argmax(prob) for prob in pred_prob]

    correct = 0
    nums = len(labels)
    for i in range(nums):
        if result[i] == labels[i]:
            correct += 1

    line = "Correct: %s ACC: %s" % (correct, float(correct)/float(nums))
    print(line)
    line += '\n'

    with open('output_acc.txt', 'w+') as f:
        f.write(line)



