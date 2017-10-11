# -*- coding: utf-8 -*-

from data_process import process_user_info, process_product_info
from behavior_analysis import collect_user_product
from user_model import build_user_features

def get_all_behavior():
    behavior_1 = collect_user_product(1)
    behavior_2 = collect_user_product(2)
    behavior_3 = collect_user_product(3)
    behavior_4 = collect_user_product(4)
    behavior = [behavior_1, behavior_2, behavior_3, behavior_4]
    return behavior

def get_product_user_relation(user, behavior):
    
    """
    输入一个用户，根据输入的用户的id，
    得到一个list包含所有跟这个用户有关的商品,这个list里每一个元素是一个list，结构为
    [商品id，时间戳，商品和用户的关系（action）]
    """
    products_1 = behavior[0][user.id]
    products_2 = behavior[1][user.id]
    products_3 = behavior[2][user.id]
    products_4 = behavior[3][user.id]
    products = products_1+products_2+products_3+products_4
    print("build products list.")
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
        
def bulid_all_users_feature(all_users, products_dict, products_feature):
    """
    构建一个用户字典，key为用户id，value为特征
    """
    users_fearture = {}
    for user in all_users:
        id = user.id
        #构建的用户特征是一个长度为 的数组
        users_fearture[id] = build_user_features(user, products_dict, products_feature)
    
    return users_fearture

def get_products_feature():
    datas = process_product_info()
    product_dict = {}
    for data in datas:
        temp_id = data[0]
        product_dict[temp_id] = data[1:]

    print("build products feature.")
    return product_dict 
        
        

def build_vector(user, user_feature, product):
    pass