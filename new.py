# -*- coding: utf-8 -*-

from data_process import process_user_info, process_product_info
from behavior_analysis import collect_user_product
from user_model import build_user_features
from business import create_business_features

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
    products = []
    if user.id in behavior[0].keys():
        products_1 = behavior[0][user.id]
        products.append(products_1)
    if user.id in behavior[1].keys():
        products_2 = behavior[1][user.id]
        products.append(products_2)
    if user.id in behavior[2].keys():
        products_3 = behavior[2][user.id]
        products.append(products_3)
    if user.id in behavior[3].keys():
        products_4 = behavior[3][user.id]
        products.append(products_4)
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
    print("build products list")
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

def build_product_input(products, behavior):
    favorite = behavior[1]
    bought = behavior[3]
    prod_feature = products.copy()
    shop_bought,shop_favorite,brand_bought,brand_favorite = create_business_features(products, bought, favorite)
    for product_id in prod_feature.keys():
        prod_feature[product_id].pop()
        shop_id = prod_feature[product_id][0]
        brand_id = prod_feature[product_id][1]
        prod_feature[product_id][0] = shop_bought[shop_id]
        prod_feature.insert(1, shop_favorite[shop_id])
        prod_feature[product_id][2] = brand_bought[brand_id]
        prod_feature.insert(3, brand_favorite[brand_id])
        
    return prod_feature
        

def build_vector(user, user_feature, product):
    pass