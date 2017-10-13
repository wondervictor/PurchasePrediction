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
from data_process import load_dataset, process_user_behaviors
from output import output_result

USER_VECTOR_SIZE = 8
PRODUCT_VECTOR_SIZE = 6

class NeuralNetwork(nn.Module):
    
    def __init__(self):

        super(NeuralNetwork, self).__init__()

        # self.user_merchandise_layer = nn.Linear(512, 512)
        self.user_self_layer = nn.Linear(USER_VECTOR_SIZE, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        # self.merchandise_description_layer = nn.Linear(512, 512)
        self.merchandise_self_layer = nn.Linear(PRODUCT_VECTOR_SIZE, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.user_layer = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.merchandise_layer = nn.Linear(64, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.hidden_layer_1 = nn.Linear(256, 256)
        
        self.hidden_layer_2 = nn.Linear(256, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 2)

    def forward(self, product_vector, user_vector):

        user_self = F.leaky_relu(self.user_self_layer(user_vector), negative_slope=0.2)
        user_self = self.bn1(user_self)

        merchandise_self = F.leaky_relu(self.merchandise_self_layer(product_vector), negative_slope=0.2)
        merchandise_self = self.bn2(merchandise_self)

        user_layer = F.leaky_relu(
            self.user_layer(user_self),
            negative_slope=0.2
        )
        user_layer = self.bn3(user_layer)

        merchandise_layer = F.leaky_relu(
            self.merchandise_layer(merchandise_self),
            negative_slope=0.2
        )
        merchandise_layer = self.bn4(merchandise_layer)

        hidden = F.leaky_relu(
            self.hidden_layer_1(torch.cat([user_layer, merchandise_layer], dim=1)),
            negative_slope=0.2
        )

        hidden = F.leaky_relu(
            self.hidden_layer_2(hidden),
            negative_slope=0.2
        )
        hidden = self.bn5(hidden)

        # hidden = F.leaky_relu(
        #     self.hidden_layer_2(hidden),
        #     negative_slope=0.2
        # )

        # hidden = F.leaky_relu(
        #     self.hidden_layer_3(hidden),
        #     negative_slope=0.2
        # )

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


def output(network, user_self_vectors, product_self_vectors):
    user_self_vector = Variable(torch.FloatTensor(user_self_vectors)).unsqueeze(0)
    product_self_vector = Variable(torch.FloatTensor(product_self_vectors)).unsqueeze(0)

    prob = network(
        product_vector=product_self_vector,
        user_vector=user_self_vector
        # user_desc_vector=user_desc_vector,
        # product_desc_vector=product_desc_vector
    )

    return prob


def train(trainset, batch_size, epoch, user_feature, product_feature):

    learning_rate = 0.0005

    network = NeuralNetwork()

    training_optimizer = optimizer.Adam(lr=learning_rate, params=network.parameters())

    loss_criterion = torch.nn.CrossEntropyLoss()
    
    count = 0
    loss_ave = 0
    for i in range(epoch):
        iters = 0
        temp = 0
        epoch_loss = 0
        for j in range(len(trainset)/batch_size):
            temp += 1
            iters += 1
            user_self_vector = []
            product_self_vector = []
            labels = []
        
            for p in range(batch_size):
                person_id = trainset[p+j][0]
                product_id = trainset[p+j][1]
                user_self_vector.append(user_feature[person_id][:11])
                product_self_vector.append(product_feature[product_id][:10])
                labels.append(trainset[p+j][-1])

            user_self_vector = Variable(torch.FloatTensor(user_self_vector))
            product_self_vector = Variable(torch.FloatTensor(product_self_vector))

            prob = network(
                product_vector=product_self_vector,
                user_vector=user_self_vector
            )

            labels = Variable(torch.LongTensor(labels))
            loss = loss_criterion(prob, labels)
            loss_ave += loss.data[0]
            epoch_loss += loss.data[0]
            if iters%100 == 0:
                print("Epoch: %s Sample: %s: Loss: %s" % (i, iters, loss_ave/100.0))
                loss_ave = 0

            training_optimizer.zero_grad()
            loss.backward()
            # for param in network.parameters():
            #     print(param.grad)
            if epoch == 15:
                for param_group in training_optimizer.param_groups:
                    learning_rate = learning_rate/5.0
                    param_group['lr'] = learning_rate
            training_optimizer.step()
        print("loss: %s", epoch_loss/float(temp))

        save_model(network, './model_param/deep1_neural_network_param_%s' % i)


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
            product_self_vector
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
            product_self_vector
        )
        prob = prob.squeeze(0)
        prob = prob.data.numpy()
        print(prob)
        if prob[0] < prob[1]:
            result.append((person_id, product_id))

    return result


if __name__ == "__main__":
    data1 = open("data/product_dict1.pkl","rb")
    data2 = open("data/user_dict1.pkl","rb")
    product_dict = pickle.load(data1)
    user_dict = pickle.load(data2)
    data1.close()
    data2.close()

    predicted = process_user_behaviors('data/predict.txt')

    result = predict(predicted, user_feature=user_dict, product_feature=product_dict, model_path='./model_param/deep1_neural_network_param_45')
    # testset = load_dataset('data/test_data.txt')

    output_result(result)
