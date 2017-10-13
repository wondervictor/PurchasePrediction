# -*- coding: utf-8 -*-

"""



"""
import pickle
import torch
import torch.nn as torch_nn
from torch.autograd import Variable
import numpy as np
from model.network import train, test
from data_process import load_dataset
import evaluation
from product_analysis import gen_product_embedding


DICT_DIM = 53900
OUTPUT_DIM = 256

def generate_user_embedding(user_dict):

    embedding = torch_nn.Embedding(DICT_DIM, OUTPUT_DIM)

    def gen(input):
        nums = len(input)
        input = Variable(torch.LongTensor(input))
        result_vector = Variable(torch.zeros((1, OUTPUT_DIM)))
        if nums == 0:
            return result_vector.data.numpy()

        output = embedding(input).data.numpy()
        result = np.sum(output, axis=0) / nums
        return result

    for key in user_dict.keys():
        #print(key, user_dict[key])
        describe = user_dict[key][8]
        #print(len(user_dict[key]))
        describe_embedding = gen(describe)
        user_dict[key][8] = describe_embedding

    return user_dict


def generate_product_embedding(product_dict):

    embedding = torch_nn.Embedding(DICT_DIM, OUTPUT_DIM)

    def gen(input):
        nums = len(input)
        input = Variable(torch.LongTensor(input))
        result_vector = Variable(torch.zeros((1, OUTPUT_DIM)))
        if nums == 0:
            return result_vector.data.numpy()

        output = embedding(input).data.numpy()
        result = np.sum(output, axis=0) / nums
        return result

    for key in product_dict.keys():

        describe = product_dict[key][6]
        describe_embedding = gen(describe)

        product_dict[key][6] = describe_embedding

    return product_dict




def load_dict():

    """
    处理训练数据
    :return:
    """
    # 构建一个用户特征dict，这个特征将会作为模型的输入，将所有用户的特征构建成一个key为用户id的dict返回
    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    product_file = open("data/product_dict.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()
    print("Finished Loading Product Dict")

    return user_dict, product_dict


def prepare_training_data():

    trainset = load_dataset('data/train_data.txt')
    testset = load_dataset('data/test_data.txt')

    print("Finished Loading Dataset")

    user_file = open("data/user_dict_embedding.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    product_file = open("data/product_dict_embedding.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()
    print("Finished Loading Product Dict")

    return trainset, testset, user_dict, product_dict


def prepare_dict():


    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    # product_file = open("data/product_dict.pkl","rb")
    # product_dict = pickle.load(product_file)
    # product_file.close()
    # print("Finished Loading Product Dict")

    #product_dict = generate_product_embedding(product_dict)
    user_dict = generate_user_embedding(user_dict)

    #print(product_dict[product_dict.keys[0]])


    user_file = open("data/user_dict_embedding.pkl",'wb')
    pickle.dump(user_dict, user_file)
    user_file.close()
    #
    # user_file = open("data/product_dict_embedding.pkl",'wb')
    # pickle.dump(product_dict, user_file)
    # user_file.close()
    print("Saved")


if __name__ == '__main__':

    # prepare_dict()

    trainset, testset, user_dict, product_dict = prepare_training_data()

    # print(user_dict[user_dict.keys()[1]])
    # print(product_dict[product_dict.keys()[1]])
    #for i in product_dict.keys()[185153:185185]:
    #    print(product_dict[i])
    #print(len(trainset))
    eva =  evaluation.evaluate
    train(trainset, batch_size=32, epoch=10, user_feature=user_dict, product_feature=product_dict, test_set=testset, evaluate=eva)
    #labels, result = test(testset,user_feature=user_dict, product_feature=product_dict, model_path='model_param/neural_network_param_2')
    #evaluation.evaluate(result, labels)


