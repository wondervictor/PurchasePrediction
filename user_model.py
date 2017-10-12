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
    price = []
    product_bought = []
    freq = 0
    for product in products:
        freq += 1
        product_id = product[0]
        product_data = feature_dict[product_id]
        discribes = product_data[4]
        all_discribes += discribes
        # if product[1] == 4:
        #     product_bought.append(product)
        #     price.append(product_data[3])
    # freq = len(product_bought)
        price.append(product_data[3])
    if freq == 0:
        return 0
    ave_price = sum(price)/float(freq)
    max_price = max(price)
    min_price = min(price)
    feature = [ave_price, max_price, min_price, freq]
    #discribe_vector = gen_product_embedding(all_discribes)
    feature.append(all_discribes)


    return feature


    # temp_vector = []
    # for product in products:
    #     temp_vector.append(feature_dict[product[0]])

    # feature = []
    # return feature


def build_user_features(user, products_dict, products_feature):

    """
    用户特征: 商品特征+固有特征 [商品成分分析] + [购物等级、贫富、孩子年龄，孩子性别]
    :return:
    """
    products = products_dict[user.id]
    #用户自己的信息构建的特征
    if user.has_baby == False:
        user.baby_age, user.baby_gender, user.has_baby = 0, 0, 0
    else:
        user.has_baby = 1   
    user_feature = [user.rank, user.has_baby, user.baby_age, user.baby_gender]
    
    #根据用户买的商品的信息作为特征
    feature_from_product = extract_features_from_product(products, products_feature)
    if feature_from_product == 0:
        return []
    feature = user_feature + feature_from_product
    return feature



