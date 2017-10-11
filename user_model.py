# -*- coding: utf-8 -*-

"""
User Model
"""

from data_process import User, process_product_info
from product_analysis import gen_product_embedding


def extract_features_from_product(products, feature_dict):
    """
    从商品中提取出人物特征：

    1. 综合特征(包含用户喜好、倾向)
    2. 用户消费水平
    3. 用户购物频率
    :param products:
    :return:
    """
    all_discribes = []
    for product in products:
        product_id = product[0]
        product_data = feature_dict[product_id]
        discribes = product_data[5]
        all_discribes += discribes
    discribe_vector = gen_product_embedding(all_discribes)

    
    # temp_vector = []
    # for product in products:
    #     temp_vector.append(feature_dict[product[0]])

    feature = []
    return feature


def build_user_features(user, products_dict, products_feature):

    """
    用户特征: 商品特征+固有特征 [商品成分分析] + [购物等级、贫富、孩子年龄，孩子性别]
    :return:
    """
    products = products_dict[user.id]
    #用户自己的信息构建的特征
    
    user_feature = [user.id, user.rank, user.hasbaby, user.baby_age, user.baby_gender]
    


    #根据用户买的商品的信息作为特征
    feature_from_product = extract_features_from_product(products, products_feature)
    feature = user_feature + feature_from_product
    print("build users feature")
    return user_feature



