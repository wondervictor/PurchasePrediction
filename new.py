# -*- coding: utf-8 -*-

from data_process import process_user_info
from behavior_analysis import collect_user_product
from user_model import build_user_features

def get_all_behavior():
    behavior_1 = collect_user_product(1)
    behavior_2 = collect_user_product(2)
    behavior_3 = collect_user_product(3)
    behavior_4 = collect_user_product(4)
    behavior = [behavior_1, behavior_2, behavior_3, behavior_4]
    return behavior

def get_products_by_user(id, products_dict):
    """
    提取跟这个用户有关的商品，这些商品将会被输入进模型中，格式为[product_id, action]
    """
    prod_by_user = []
    products = products_dict[id]
    product_1 = products[0][0]#浏览的商品
    product_2 = products[1][0]#收藏
    product_3 = products[2][0]#加购
    product_4 = products[3][0]#购买
    for product in product_1:
        temp = [product, 1]
        prod_by_user.append(temp)
    for product in product_2:
        temp = [product, 2]
        prod_by_user.append(temp)
    for product in product_3:
        temp = [product, 3]
        prod_by_user.append(temp)
    for product in product_4:
        temp = [product, 4]
        prod_by_user.append(temp)
        
    return prod_by_user


def get_product_user_relation(user, behavior):
    
    """
    输入一个用户，根据输入的用户的id，
    得到一个 list[跟这个用户有关商品，时间戳，商品和用户的关系（action）]
    """
    products_1 = behavior[0][user.id]
    products_2 = behavior[1][user.id]
    products_3 = behavior[2][user.id]
    products_4 = behavior[3][user.id]
    products = [products_1, products_2, products_3, products_4]
    return products


def build_product_dict(all_users, behavior):
    """
    构建一个字典，每个key是用户id，每个value是与这个id相关的商品的list，list里的元素是商品特征
    """
    products_by_user = {}
    for user in all_users:
        id = user.id
        products = get_product_user_relation(user, behavior)
        products_by_user[id]=products
    return products_by_user
        
def bulid_all_users_feature(all_users, products_dict):
    """
    构建一个用户字典，key为用户id，value为特征
    """
    users_fearture = {}
    for user in all_users:
        id = user.id
        #构建的用户特征是一个长度为 的数组
        users_fearture[id] = build_user_features(user, products_dict)
    
    return users_fearture
