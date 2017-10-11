# -*- coding: utf-8 -*-

from scipy.spatial.distance import cosine
from behavior_analysis import process_user_behaviors, filter_time, write_to_file, date_to_timestamp


def cf_product_based(all_products, user_product):

    """
    Collaborative Filtering: Product Based (基于商品的协同过滤推荐)
    :param all_products: 所有商品
    :param user_product: 用户已购买或者加购的商品
    :return:
    """

    pass


def cf_person_based(rank):
    """
    基于人的协同过滤
    :rank: 选取rank名相似用户进行推荐
    :return:
    """

    pass


def gen_recommendation_data():

    predicted_data = []

    action_1 = process_user_behaviors('data/behaviors/action_1.txt')

    start_time = "2017-7-20 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_1,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    action_2 = process_user_behaviors('data/behaviors/action_2.txt')

    start_time = "2017-7-19 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_2, date_to_timestamp(start_time), end_time)
    print(len(predicted_data))

    action_3 = process_user_behaviors('data/behaviors/action_3.txt')

    start_time = "2017-7-18 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_3,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    action_4 = process_user_behaviors('data/behaviors/action_4.txt')

    start_time = "2017-7-22 00:00:00"
    end_time = -1

    predicted_data += filter_time(action_4,  date_to_timestamp(start_time), end_time)

    print(len(predicted_data))

    write_to_file(predicted_data, 'data/predict.txt')


if __name__ == '__main__':

    gen_recommendation_data()






