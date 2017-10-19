# -*- coding: utf-8 -*-

from sklearn import tree
from sklearn.externals import joblib


def train_tree(train_set, user_dict, product_dict):

    train_data = []
    train_labels = []
    counter = 0
    for i in range(len(train_set)):
        person_id = train_set[i][0]
        product_id = train_set[i][1]
        if product_id not in product_dict or person_id not in user_dict:
            counter += 1
            continue
        train_labels.append(train_set[i][-1])
        train_data.append(user_dict[person_id][:-1]+product_dict[product_id][:-1])
    print("Generate Traing Data")

    clf = tree.DecisionTreeClassifier()

    clf.fit(train_data, train_labels)

    joblib.dump(clf, 'model_param/tree.model')


def test_tree(test_set, user_dict, product_dict):

    labels = []
    preds = []

    model = joblib.load('model_param/tree.model')

    for i in range(len(test_set)):
        person_id = test_set[i][0]
        product_id = test_set[i][1]
        if product_id not in product_dict or person_id not in user_dict:
            continue
        user = user_dict[person_id][:-1] + product_dict[product_id][:-1]
        label = test_set[i][-1]
        prob_y = int(model.predict([user])[0])
        if label == 1:
            labels.append((person_id, product_id))
        if prob_y == 1:
            preds.append((person_id, product_id))
    return labels, preds


def predict_tree(predict_set, user_dict, product_dict):

    model = joblib.load('model_param/tree.model')

    nums = len(predict_set)

    result = []
    counter = 0
    for i in range(nums):
        person_id = predict_set[i][0]
        product_id = predict_set[i][1]
        if product_id not in product_dict:
            print('-', product_id)
            continue
        if person_id not in user_dict:
            print('+',person_id)
            continue
        user = user_dict[person_id][:-1] + product_dict[product_id][:-1]
        prob = int(model.predict([user])[0])
        if prob == 1:
            result.append([person_id, product_id])
    print(counter)
    return result
