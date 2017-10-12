# -*- coding: utf-8 -*-

from data_process import process_user_info, process_product_info
from behavior_analysis import collect_user_product
from user_model import build_user_features
from business import create_business_features
from copy import deepcopy

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
    try:
        products_1 = []
        for id_num in behavior[0][user.id]:
            products_1.append([id_num,1])
        products += products_1
    except:
        pass    
    try:
        products_2 = []
        for id_num in behavior[1][user.id]:
            products_2.append([id_num,2])
        products += products_2
    except:
        pass 
    try:
        products_3 = []
        for id_num in behavior[2][user.id]:
            products_3.append([id_num,3])
        products += products_3
    except:
        pass 
    try:
        products_4= []
        for id_num in behavior[3][user.id]:
            products_4.append([id_num,4])
        products += products_4
    except:
        pass 
    return products


def build_product_dict(all_users, behavior):
    """
    构建一个字典，每个key是用户id，每个value是与这个id相关的商品的list，list里的元素是商品特征
    """
    print("start build products dict")
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
    i = 0
    for user in all_users:
        id = user.id
        i += 1
        #构建的用户特征是一个长度为 的数组
        users_fearture[id] = build_user_features(user, products_dict, products_feature)
        print(i)

    return users_fearture


def get_products_feature():
    print("start build products feature")
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
    prod_feature = {}
    for product in products.keys():
        prod_feature[product] = deepcopy(products[product])
    shop_bought,shop_favorite,brand_bought,brand_favorite = create_business_features(products, bought, favorite)
    print(len(shop_bought),len(shop_favorite),len(brand_bought),len(brand_favorite))
    i = 0
    for product_id in prod_feature.keys():
        # prod_feature[product_id].pop()
        shop_id = prod_feature[product_id][0]
        brand_id = prod_feature[product_id][1]
        try:
            prod_feature[product_id][0] = shop_bought[shop_id]
        except:
            try:
                print("shop_bought[shop_id]: ", shop_bought[shop_id])
            except:
                print("bought, shop_id: ", shop_id)
                i += 1
            prod_feature[product_id][0] = 0        
        try:
            prod_feature[product_id].insert(1, shop_favorite[shop_id])
        except:
            try:
                print("shop_favorite[shop_id]: ", shop_favorite[shop_id])
            except:
                print("favorite, shop_id: ", shop_id)
                i += 1
            prod_feature[product_id].insert(1, 0)    
        try:
            prod_feature[product_id][2] = brand_bought[brand_id]
        except:
            prod_feature[product_id][2] = 0
        try:
            prod_feature[product_id].insert(3, brand_favorite[brand_id])
        except:
            prod_feature[product_id].insert(3, 0)
    print(i)
    return prod_feature
        

def build_vector(user, user_feature, product):
    pass