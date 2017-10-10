# -*- coding: utf-8 -*-

import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


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
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())


def get_similarity():
    """
    利用TD-IDF和距离度量相似度
    :return:
    """
    pass


def gen_tfidf_vector(sentences):
    """
    生成TF-IDF向量(写入*.npy)
    :param sentences: {key: 商品ID, value:商品分词描述}
    :return:
    """








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
