# -*- coding: utf-8 -*-
# import warnings
# warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import logging
import os.path
import sys
import multiprocessing
# from gensim.models.word2vec import LineSentence
# from gensim.models import Word2Vec
import numpy as np
import math
import output
from scipy.spatial.distance import cosine
from data_process import process_user_info
from behavior_analysis import process_user_behaviors, filter_time, write_to_file, date_to_timestamp, get_user_products
from product_analysis import get_similarity



# def similarity_embedding(embed_a, embed_b):
#     """
#     获得相似度 / embedding 特征
#     :param embed_a:
#     :param embed_b:
#     :return:
#     """
#     if __name__ == '__main__':
#         program=os.path.basename(sys.argv[0])
#         logger=logging.getLogger(program)
#         logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
#         logger.info('running %s'%' '.join(sys.argv))
#         fdir='Desktop/data/'
#         inp=fdir+'data.txt'
#         outp1=fdir+'text.model'
#         outp2=fdir+'text.vector'
#         model=Word2Vec(LineSentence(inp),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())
#         model.save(outp1)
#         mdoel.wv.save_word2vec_format(outp2,binary=False)
#
#     '''
#         calculate cosine
#
#     '''
#
#     return get_similarity(embed_a, embed_b)


def similarity_person_self(self_a, self_b):
    """
    固有特征
    :param self_a:
    :param self_b:
    :return:
    """
    self_a = np.array(self_a)
    self_b = np.array(self_b)
    return get_similarity(self_a, self_b)


def cf_product_based(all_products, user_product):

    """
    Collaborative Filtering: Product Based (基于商品的协同过滤推荐)
    :param all_products: 所有商品
    :param user_product: 用户已购买或者加购的商品
    :return:
    """

    pass


def cf_person_based(k=4, num_user=10):
    """
    基于人的协同过滤
    :param k k名相似
    :return:
    """

    user_purchased = get_user_products()

    product_user_relation = dict()

    for user, item in user_purchased.items():
        for i in item.keys():
            if i not in product_user_relation:
                product_user_relation[i] = set()
            product_user_relation[i].add(user)

    C = dict()
    N = dict()
    for i, users in product_user_relation.items():

        for u in users:
            N.setdefault(u, 0)
            N[u] += 1
            C.setdefault(u, {})

            for v in users:
                if u == v:
                    continue
                C[u].setdefault(v, 0)
                C[u][v] += 1
    W = dict()

    for u, relate_users in C.items():
        W.setdefault(u, {})
        for v, cuv in relate_users.items():
            W[u][v] = cuv / math.sqrt(N[u]*N[v])

    recommend_list = []

    print("Starting to recommend")

    for user in user_purchased.keys():
        rank = dict()
        action_item = user_purchased[user].keys()
        for v, wuv in sorted(W[user].items(), key=lambda x: x[1], reverse=True)[0:k]:

            for i, rvi in user_purchased[v].items():
                if i in action_item:
                    continue
                rank.setdefault(i, 0)
                rank[i] += wuv*rvi

        recom_dict = dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:num_user])

        recom_products = [(user, product) for product in recom_dict.keys()]

        recommend_list += recom_products

    print(len(recommend_list))
    output.output_result_path(recommend_list, 'data/recom.txt')


def gen_recommendation_data1():

    predicted_data = []

    action_1 = process_user_behaviors('data/behaviors/action_1.txt')

    start_time = "2017-8-23 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_1,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    action_2 = process_user_behaviors('data/behaviors/action_2.txt')

    start_time = "2017-8-19 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_2, date_to_timestamp(start_time), end_time)
    print(len(predicted_data))

    action_3 = process_user_behaviors('data/behaviors/action_3.txt')

    start_time = "2017-8-18 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_3,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    action_4 = process_user_behaviors('data/behaviors/action_4.txt')

    start_time = "2017-8-22 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_4,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    write_to_file(predicted_data, 'data/predict.txt')


if __name__ == '__main__':

    #gen_recommendation_data()

    cf_person_based()
