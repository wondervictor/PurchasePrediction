# -*- coding: utf-8 -*-

"""
Neural Network Model
神经网络模型

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.autograd import Variable
import cPickle as pickle
from data_process import load_dataset

USER_VECTOR_SIZE = 8
PRODUCT_VECTOR_SIZE = 6

class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()

        # self.user_merchandise_layer = nn.Linear(512, 512)
        self.user_self_layer = nn.Linear(USER_VECTOR_SIZE, 64)

        # self.merchandise_description_layer = nn.Linear(512, 512)
        self.merchandise_self_layer = nn.Linear(PRODUCT_VECTOR_SIZE, 64)

        self.user_layer = nn.Linear(64, 256)

        self.merchandise_layer = nn.Linear(64, 256)

        self.hidden_layer_1 = nn.Linear(512, 1024)
        self.hidden_layer_2 = nn.Linear(1024, 512)
        self.hidden_layer_3 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, product_vector, user_vector):

        # user_desc = F.relu(self.user_merchandise_layer(user_desc_vector))

        user_self = F.leaky_relu(self.user_self_layer(user_vector), negative_slope=0.2)
        # merchandise_desc = F.relu(self.merchandise_description_layer(product_desc_vector))
        merchandise_self = F.leaky_relu(self.merchandise_self_layer(product_vector), negative_slope=0.2)

        user_layer = F.leaky_relu(
            self.user_layer(user_self),
            negative_slope=0.2
        )

        merchandise_layer = F.leaky_relu(
            self.merchandise_layer(merchandise_self),
            negative_slope=0.2
        )

        hidden = F.leaky_relu(
            self.hidden_layer_1(torch.cat([user_layer, merchandise_layer])),
            negative_slope=0.2
        )

        hidden = F.leaky_relu(
            self.hidden_layer_2(hidden),
            negative_slope=0.2
        )

        hidden = F.leaky_relu(
            self.hidden_layer_3(hidden),
            negative_slope=0.2
        )

        output = F.softmax(
            self.out(hidden)
        )

        return output

class SimpleNetwork(nn.Module):
    
    def __init__(self):
        super(SimpleNetwork, self).__init__()

    def forward(self):
        pass
    



def save_model(network, path):

    torch.save(network, path)
    print("Save Model to File: %s" % path)


def load_model(path):

    network = torch.load(path)

    return network


def output(network, user_self_vector, product_self_vector):
    user_self_vector = Variable(torch.FloatTensor(user_self_vector))
    # user_desc_vector = Variable(torch.FloatTensor(user_desc_vector))
    # product_desc_vector = Variable(torch.FloatTensor(product_desc_vector))
    product_self_vector = Variable(torch.FloatTensor(product_self_vector))

    prob = network(
        product_vector=product_self_vector,
        user_vector=user_self_vector
        # user_desc_vector=user_desc_vector,
        # product_desc_vector=product_desc_vector
    )

    return prob


def train(trainset, epoch, user_feature, product_feature):

    learning_rate = 0.00001

    network = NeuralNetwork()

    training_optimizer = optimizer.Adam(lr=learning_rate, params=network.parameters())

    loss_criterion = torch.nn.CrossEntropyLoss()
    
    loss_ave = 0
    for i in range(epoch):
        iters = 0

        for data in trainset:
            iters += 1
            person_id = data[0]
            product_id = data[1]
            label = data[-1]
            user_self_vector = user_feature[person_id][:-1]
            if user_feature[person_id] == []:
                continue

            product_self_vector = product_feature[product_id][:-1]

            label = Variable(torch.LongTensor([label]))

            prob = output(
                network,
                user_self_vector,
                # user_desc_vector,
                product_self_vector
                # product_desc_vector
            )
            prob = prob.unsqueeze(0)
            loss = loss_criterion(prob, label)
            loss_ave += loss.data[0]
            if iters % 1000 == 0:
                print("Sample: %s: Loss: %s" % (iters, loss_ave/1000.0))
                loss_ave = 0

            training_optimizer.zero_grad()
            loss.backward()
            training_optimizer.step()
        save_model(network, './model_param/neural_network_param_%s' % i)


def test(testset, user_feature, product_feature, model_path):

    network = load_model(model_path)

    result = []
    labels = []

    for data in testset:
        person_id = data[0]
        product_id = data[1]
        label = data[-1]
        user_self_vector = user_feature[person_id][:-1]

        product_self_vector = product_feature[product_id][:-1]

        prob = output(
            network,
            user_self_vector,
            # user_desc_vector,
            product_self_vector
            # product_desc_vector
        )

        labels.append(label)
        result.append(prob)

    return labels, result


def predict(predict_set, user_feature, product_feature, model_path):

    network = load_model(model_path)
    result = []

    for data in predict_set:
        person_id = data[0]
        product_id = data[1]

        user_self_vector = user_feature[person_id][:-1]

        product_self_vector = product_feature[product_id][:-1]

        prob = output(
            network,
            user_self_vector,
            # user_desc_vector,
            product_self_vector
            # product_desc_vector
        )

        result.append(prob)

    return result

if __name__ == "__main__":
    data1 = open("data/product_dict.pkl","rb")
    data2 = open("data/user_dict3.pkl","rb")
    product_dict = pickle.load(data1)
    user_dict = pickle.load(data2)
    data1.close()
    data2.close()

    trainset = load_dataset('data/train_data.txt')
    testset = load_dataset('data/test_data.txt')
    
    train(trainset, 30, user_dict, product_dict)