# -*- coding: utf-8 -*-

import pickle
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
from data_process import process_product_info


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


# TD-IDF分词处理
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


def get_similarity(product_a, product_b, tfidf_vector):
    """
    利用TD-IDF和距离度量相似度
    :return:
    """

    desciption_a = tfidf_vector[product_a]
    desciption_b = tfidf_vector[product_b]

    return dist_norm(desciption_a, desciption_b)


def train_tfidf_model(sentences, tfidf_model):

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
    process_words()
