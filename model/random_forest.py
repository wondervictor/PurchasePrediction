# -*- coding: utf-8 -*-

"""
Training Model: Random Forest

"""

from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=1000, max_depth=15)

