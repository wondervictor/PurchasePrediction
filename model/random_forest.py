# -*- coding: utf-8 -*-

"""
Training Model: Random Forest

"""

from sklearn.ensemble import RandomForestClassifier


class RandomForest(object):

    def __init__(self, max_depth):

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            random_state=10131,
        )

    def train(self, input, labels):

        self.model.fit(input, labels)

    def predict(self, input):

        predicted = self.model.predict(input)

        return predicted

    def save(self, path):

        pass









