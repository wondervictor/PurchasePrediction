# -*- coding: utf-8 -*-

"""



"""
import pickle
from model import network
from data_process import load_dataset



def prepare_data():

    """
    处理训练数据
    :return:
    """

    trainset = load_dataset('data/train_data.txt')
    testset = load_dataset('data/test_data.txt')

    #构建一个用户特征dict，这个特征将会作为模型的输入，将所有用户的特征构建成一个key为用户id的dict返回
    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print(len(user_dict))
    print(user_dict[user_dict.keys()[0]])

    product_file = open("data/product_dict.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()

    print(len(product_dict))
    print(product_dict[product_dict.keys()[0]])


if __name__ == '__main__':
    prepare_data()