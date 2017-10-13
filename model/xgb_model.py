# -*- coding: utf-8 -*-

"""
XGBoost for Classification
"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import


class XGBModel(object):


    def __init__(self):

        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'early_stopping_rounds': 100,
            'scale_pos_weight': 1,
            'eval_metric': 'auc',
            'gamma': 0.2,
            'max_depth': 15,
            'lambda ': 100,
            'subsample': 1,
            'colsample_bytree': 0.5,
            'min_child_weight': 5,
            'eta': 0.05,
            'seed': 2321531,
            'nthread': 2,
            'max_delta_step': 1
        }

    def train(self, X, y):

        d_train = xgb.DMatrix(data=X,label=y)
        self.xgbmodel =xgb.train(self.params, dtrain=d_train, num_boost_round=1000)


    def predict(self, X):

        d_train = xgb.DMatrix(data=X)
        prob_y = self.xgbmodel.predict(d_train)

        return prob_y

    def save_model(self):

        pass



