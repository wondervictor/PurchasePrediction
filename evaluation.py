# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
import numpy as np


def fscore(truth, pred):
    """

    :param truth: [(user_id, product_id)]
    :param pred: [(user_id, product_id)]
    :return:
    """
    truth_num = len(truth)
    pred_num = len(pred)

    truth_map = {}
    for i in range(len(truth)):
        if truth[i] not in truth_map:
            truth_map[truth[i]] = 0
        truth_map[truth[i]] += 1

    common = 0
    for i in range(len(pred)):
        if pred[i] in truth_map and pred[i] != 0:
            common += 1
            truth_map[pred[i]] -= 1

    precision = float(common)/float(pred_num)
    recall = float(common)/float(truth_num)

    f1 = 2*precision*recall/(precision+recall)

    print("Recall: %s Precision: %s F1-Score: %s" % (recall, precision, f1))
    return f1


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



