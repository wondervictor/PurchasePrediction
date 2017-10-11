# -*- coding: utf-8 -*-

from data_process import process_user_info
from behavior_analysis import collect_user_product
from new import get_all_behavior, build_product_dict, bulid_all_users_feature, get_products_by_user, build_vector, get_products_feature


if __name__ == "__main__":
    #读取用户信息，将用户信息存入User对象，将所有User对象构建一个数组
    all_users = process_user_info()
    
    #读取action文件，将所有用户行为分成四类，存入一个长度为4的数组里，每个元素是以用户id为key的字典
    behavior = get_all_behavior()
    
    #构建一个dict，每个key是用户id，内容是商品list，list中的每个元素是一个商品的id、时间戳、action，这个list会作为模型的输入
    priducts_dict = build_product_dict(all_users, behavior)
    
    #构建一个dict，每个key是商品id，value是商品特征
    products_feature = get_products_feature()

    #构建一个用户特征，这个特征将会作为模型的输入，将所有用户的特征构建成一个key为用户id的dict返回
    user_feature_dict = bulid_all_users_feature(all_users, priducts_dict)
    

    # for user in all_users:
    #     #输入到模型里的用户
    #     user_feature = user_feature_dict[user.id]
    #     #输入到模型里的商品
    #     prod_by_user = get_products_by_user(user.id, priducts_dict)
    #     for each_product in prod_by_user:
    #         #构建输入向量和label
    #         input_vector = build_vector(user, user_feature, each_product)
    #         label = 



        
        
        
    
