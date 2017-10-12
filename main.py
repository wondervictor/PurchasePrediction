# -*- coding: utf-8 -*-

"""



"""
import pickle
from model.network import train, test
from data_process import load_dataset
from product_analysis import gen_product_embedding


def generate_user_embedding(user_dict):

    for key in user_dict.keys():
        describe = user_dict[key][9]
        describe_embedding = gen_product_embedding(describe, output_dim=256)
        user_dict[key][9] = describe_embedding

    return user_dict


def generate_product_embedding(product_dict):

    for key in product_dict.keys():

        describe = product_dict[key][6]
        describe_embedding = gen_product_embedding(describe, output_dim=256)

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

    user_dict, product_dict = load_dict()
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

    prepare_dict()

    #trainset, testset, user_dict, product_dict = prepare_training_data()

    #train(trainset, epoch=10, user_feature=user_dict, product_feature=product_dict)

