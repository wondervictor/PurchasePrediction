# -*- coding: utf-8 -*-

"""
XGBoost for Classification
"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

class XGB_Model(object):


    def __init__(self):

        self.params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'early_stopping_rounds': 100,
            'scale_pos_weight': 1,
            'eval_metric': 'auc',
            'gamma': 0.2,
            'max_depth': 14,
            'lambda ': 100,
            'subsample': 1,
            'colsample_bytree': 0.5,
            'min_child_weight': 5,
            'eta': 0.001,
            'seed': 2321531,
            'nthread': 2,
            'num_class':2,
            'max_delta_step': 1
        }

    def train(self, X, y):
        d_train = xgb.DMatrix(data=X,label=y)
        self.xgbmodel =xgb.train(self.params, dtrain=d_train, num_boost_round=200)
        self.xgbmodel.save_model('./model_param/xgb.model')

    def predict(self, X):

        d_train = xgb.DMatrix(data=X)
        prob_y = self.xgbmodel.predict(d_train)

        return prob_y


def train(train_set, test_set, user_dict, product_dict):


    train_data = []
    train_labels = []
    for i in range(len(train_set)):
        person_id = train_set[i][0]
        product_id = train_set[i][1]
        train_labels.append(train_set[i][-1])
        train_data.append(user_dict[person_id][:-1]+product_dict[product_id][:-1])

    print("Generate Traing Data")
    model = XGB_Model()
    model.train(train_data, train_labels)
    print("Training")
    del train_data

    f = open('xgb_model_output.txt', 'w+')
    correct = 0
    nums = len(test_set)
    for i in range(len(test_set)):
        person_id = test_set[i][0]
        product_id = test_set[i][1]
        user = user_dict[person_id][:-1] + product_dict[product_id][:-1]
        label = test_set[i][-1]
        prob_y = int(model.predict([user])[0])
        if prob_y == label:
            correct += 1
        f.write("%s:%s\n" % (prob_y, label))
    print("Correct: %s ACC:%s" % (correct, float(correct)/float(nums)))
    f.close()


def predict(predict_set, user_dict, product_dict):

    model = xgb.Booster({'nthread':2})
    model.load_model('model_param/xgb.model')

    nums = len(predict_set)

    result = []
    for i in range(nums):
        person_id = predict_set[i][0]
        product_id = predict_set[i][1]
        user = user_dict[person_id][:-1] + product_dict[product_id][:-1]
        x = xgb.DMatrix([user])
        prob = int(model.predict(x)[0])

        if prob == 1:

            result.append([person_id, product_id])

    return result









