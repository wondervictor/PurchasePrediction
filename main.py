# -*- coding: utf-8 -*-

"""



"""
import pickle
from model.network import train, test
from data_process import load_dataset
from product_analysis import gen_product_embedding


def prepare_data():

    """
    处理训练数据
    :return:
    """

    trainset = load_dataset('data/train_data.txt')
    testset = load_dataset('data/test_data.txt')

    print("Finished Loading Dataset")

    # 构建一个用户特征dict，这个特征将会作为模型的输入，将所有用户的特征构建成一个key为用户id的dict返回
    user_file = open("data/user_dict.pkl",'rb')
    user_dict = pickle.load(user_file)
    user_file.close()
    print("Finished Loading User Dict")

    product_file = open("data/product_dict.pkl","rb")
    product_dict = pickle.load(product_file)
    product_file.close()
    print("Finished Loading Product Dict")

    return trainset, testset, user_dict, product_dict


if __name__ == '__main__':
    trainset, testset, user_dict, product_dict = prepare_data()
    embedding_method = gen_product_embedding
    print("Start to train")
    train(trainset, epoch=10, user_feature=user_dict, product_feature=product_dict, embedding=embedding_method)

