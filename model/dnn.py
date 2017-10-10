# -*- coding: utf-8 -*-

"""
Neural Network Model
神经网络模型

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


USER_VECTOR_SIZE=10
PRODUCT_VECTOR_SIZE=10


class NeuralNetwork(nn.Module):

    def __init__(self, dict_size):

        super(NeuralNetwork, self).__init__()

        self.embedding = nn.Embedding(dict_size, 512)
        self.user_layer = nn.Linear(USER_VECTOR_SIZE, 256)
        self.product_layer = nn.Linear(PRODUCT_VECTOR_SIZE, 512)



    def forward(self, product_vector, user_vector):

        pass
