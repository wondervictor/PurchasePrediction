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

        self.user_merchandise_layer = nn.Linear(512, 512)
        self.user_self_layer = nn.Linear(USER_VECTOR_SIZE, 128)

        self.merchandise_description_layer = nn.Linear(512, 512)
        self.merchandise_self_layer = nn.Linear(PRODUCT_VECTOR_SIZE, 128)

        self.user_layer = nn.Linear(640, 512)

        self.merchandise_layer = nn.Linear(640, 512)

        self.hidden_layer_1 = nn.Linear(1024, 1024)
        self.hidden_layer_2 = nn.Linear(1024, 512)
        self.hidden_layer_3 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, product_vector, user_vector, user_desc_vector, product_desc_vector):

        user_desc = F.relu(self.user_merchandise_layer(user_desc_vector))
        user_self = F.relu(self.user_self_layer(user_vector))
        merchandise_desc = F.relu(self.merchandise_description_layer(product_desc_vector))
        merchandise_self = F.relu(self.merchandise_self_layer(product_vector))

        user_layer = F.leaky_relu(
            self.user_layer(torch.cat([user_desc, user_self])),
            negative_slope=0.2
        )

        merchandise_layer = F.leaky_relu(
            self.merchandise_layer(torch.cat([merchandise_desc, merchandise_self])),
            negative_slope=0.2
        )

        hidden = F.leaky_relu(
            self.hidden_layer_1(torch.cat([user_layer, merchandise_layer])),
            negative_slope=0.5
        )

        hidden = F.leaky_relu(
            self.hidden_layer_2(hidden),
            negative_slope=0.5
        )

        hidden = F.leaky_relu(
            self.hidden_layer_3(hidden),
            negative_slope=0.5
        )

        output = F.softmax(
            self.out(hidden)
        )

        return output

