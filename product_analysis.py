# -*- coding: utf-8 -*-

import pickle
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
import torch
import torch.nn as torch_nn
from torch.autograd import Variable
from data_process import process_product_info

"""
采用Embedding模型进行特征提取
IF-IDF模型废除
"""


def get_same_class_product(class_id, product_data):
    """
    获得同类商品
    :param class_id: 商品类别
    :param product_data: 商品数据
    :return:
    """
    def _filter_(x):
        if x[2] == class_id:
            return True
        else:
            return False

    class_data = filter(_filter_, product_data)

    return class_data


def dist_norm(v1, v2):
    """
    距离度量
    :param v1:
    :param v2:
    :return:
    """
    v1_normalized = v1/sp.linalg.norm(v1)
    v2_normalized = v2/sp.linalg.norm(v2)
    print(v1_normalized.shape, v2_normalized.shape)
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta)


def get_similarity(product_a, product_b):
    """
    利用距离度量相似度
    :return:
    """

    return dist_norm(product_a, product_b)


def train_tfidf_model(sentences, tfidf_model):
    """
    训练TF-IDF模型

    :param sentences:
    :param tfidf_model:
    :return:
    """
    tf_vector = tfidf_model.fit_transform(sentences)
    print(tf_vector.toarray().shape)


def gen_tfidf_vector(sentences, tfidf_model):
    """
    生成TF-IDF向量(写入*.npy)
    :param sentences: {key: 商品ID, value:商品分词描述}
    :return:
    """
    tf_vector = tfidf_model.transform(sentences)

    return tf_vector.toarray()


def process_words():
    """
    利用 TF-IDF计算相似度
    :return:
    """
    product_info = process_product_info()

    sentences = [x[5][1:] for x in product_info]

    stops = ['~', '@', ',', '【', '】', '#', '$', '%', '&', '!', '+', '-', '/', '*',
                ':', ';', '?', '{', '}', '¥', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '(', ')', '<', '>', '=', '|', 'a', '《', '》', '。', '的', '地', '得']

    transformer = TfidfVectorizer(min_df=2, stop_words=stops)

    nums_examples = len(product_info)

    print(nums_examples)

    batch_size = 60000

    for i in range(nums_examples/batch_size-1):
        train_tfidf_model(sentences[batch_size*i:batch_size*i+batch_size], transformer)

    tfidf_values = {}

    for ele in product_info:
        pro_id = ele[0]
        tfidf_values[pro_id] = gen_tfidf_vector([ele[5][1:]], transformer)[0]

    f = open('model_param/tfidf_value.pkl', 'wb')
    pickle.dump(tfidf_values, f)
    f.close()
    print("Saved to files")


def gen_all_product_embedding_info(dict_dim, output_dim):

    embedding = torch_nn.Embedding(dict_dim, output_dim)

    # 线性求和
    def gen_embedding_vector(input, nums):
        result_vector = Variable(torch.zeros((1, output_dim)))
        if nums == 0:
            return result_vector

        output = embedding(input)
        for i in range(nums):
            result_vector += output[i]

        return result_vector

    products = process_product_info()

    description_dict = {}

    for product in products:

        product_id = product[0]
        description = product[5]
        nums_ = len(description)
        description = Variable(torch.LongTensor(description))

        output_vector = gen_embedding_vector(description, nums_)

        output_vector = output_vector.data.numpy()
        description_dict[product_id] = output_vector

    return description_dict


def gen_product_embedding(input, dict_dim=53900, output_dim=512):
    """
    生成词向量
    :param input:输入为描述序列
    :param dict_dim:
    :param output_dim:
    :return:
    """

    embedding = torch_nn.Embedding(dict_dim, output_dim)
    nums = len(input)
    input = Variable(torch.LongTensor(input))
    result_vector = Variable(torch.zeros((1, output_dim)))
    if nums == 0:
        return result_vector

    output = embedding(input)
    for i in range(nums):
        result_vector += output[i]

    return result_vector


# 构建特征
def construct_product_features():
    """
    构建商品特征：
    1. 价格
    2. 描述特征
    3. 店铺评价
    4. 商品是否浏览过、收藏过、加购过
    :return:
    """
    pass


if __name__ == '__main__':

    embedding_dict = gen_all_product_embedding_info(53900, 1000)

    print(len(embedding_dict))


