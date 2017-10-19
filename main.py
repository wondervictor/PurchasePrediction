# -*- coding: utf-8 -*-

"""
Main
"""
import pickle
import torch
import torch.nn as torch_nn
from torch.autograd import Variable
import numpy as np
from model.network import train, test
from data_process import load_dataset
import evaluation
from model import xgb_model
from output import output_result
from product_analysis import gen_product_embedding
from model import svm, random_forest, descision_tree


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
    """
    准备训练数据
    """

    trainset = load_dataset('data/train_data.txt')
    testset_recom = load_dataset('data/test_predict_recom_data.txt')
    test_result = load_dataset('data/test_predict_data.txt')

    print("Finished Loading Dataset")

    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    product_file = open("data/product_dict.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()
    print("Finished Loading Product Dict")

    return trainset, testset_recom, test_result, product_dict, user_dict


def prepare_dict():
    """
    Generate Embedding-User from User Dict 
    """

    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    product_file = open("data/product_dict.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()
    print("Finished Loading Product Dict")

    product_dict = generate_product_embedding(product_dict)
    user_dict = generate_user_embedding(user_dict)



    user_file = open("data/user_dict_embedding.pkl",'wb')
    pickle.dump(user_dict, user_file)
    user_file.close()
    
    user_file = open("data/product_dict_embedding.pkl",'wb')
    pickle.dump(product_dict, user_file)
    user_file.close()
    print("Saved")


if __name__ == '__main__':

    trainset, testset_recom, test_result, product_dict, user_dict = prepare_training_data()
    predict_set = load_dataset('data/predict.txt')
    predict_recom_set = load_dataset('data/recom.txt')

    predict_set = predict_set + predict_recom_set

    # 使用SVM训练
    svm.train_svm(trainset, user_dict, product_dict)
    labels, preds = svm.test(testset_recom, user_dict, product_dict)
    evaluation.fscore(labels, preds)
    result = svm.predict(predict_set, user_dict, product_dict)
    output_result(result)




