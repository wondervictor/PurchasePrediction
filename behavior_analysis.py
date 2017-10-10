# -*- coding: utf-8 -*-

import time
import datetime
from data_process import process_user_behaviors
import output


# date convension

def date_to_timestamp(date):
    """
    日期转时间戳
    :param date:
    :return:
    """
    if len(date) == 0:
        return -1
    pattern = '%Y-%m-%d %H:%M:%S'
    strptime = time.strptime(date, pattern)
    timestamp = time.mktime(strptime)
    return timestamp


def timestamp_to_date(timestamp):
    """
    时间戳转日期
    :param timestamp:
    :return:
    """
    date_ = datetime.datetime.fromtimestamp(timestamp)
    date_str = date_.strftime('%Y-%m-%d %H:%M:%S')
    return date_str


def filter_user_action(action, data):
    """
    :param action: 1-浏览 2-收藏 3-加购 4-购买
    """

    def _filter_(element):
        if element[-1] == action:
            return True
        else:
            return False

    new_data = filter(_filter_, data)

    return new_data


def filter_time(data, start_time, end_time=-1):

    """
    filter with timestamp
    :param start_time: start time/Unix Timestamp
    :param end_time: end time/Unix Timestamp
    """

    def _filter_(element):

        if element[2] >= start_time and (element[2] <= end_time or end_time == -1):
            return True
        else:
            return False

    new_data = filter(_filter_, data)

    return new_data


def write_to_file(behaviors, filepath, timestamp=False):
    """
    将用户行为写入文件
    :param behaviors:
    :param filepath:
    :param timestamp:
    :return:
    """
    with open(filepath, 'w+') as f:

        for element in behaviors:
            line = '\t'.join(['%s' % x for x in element])
            line += '\n'
            f.write(line)
    print("Saved to file: %s" % filepath)


def remove_purchased(purchased, shopping_cart):

    def _filter(element):
        for pur_ele in purchased:
            if element[0] == pur_ele[0] and element[1] == pur_ele[1] and element[2] < pur_ele[2]:
                return True

        return False

    _shopping = filter(_filter, shopping_cart)

    return _shopping


def split_action_behaviors():
    """
    根据行为action拆分数据
    :return:
    """
    behaviors = process_user_behaviors()
    action_one_behavior = filter_user_action(1, behaviors)
    action_two_behavior = filter_user_action(2, behaviors)
    action_three_behavior = filter_user_action(3, behaviors)
    action_four_behavior = filter_user_action(4, behaviors)
    del behaviors
    dir_path = './data/behaviors/'
    write_to_file(action_one_behavior, dir_path+'action_1.txt')
    write_to_file(action_two_behavior, dir_path+'action_2.txt')
    write_to_file(action_three_behavior, dir_path+'action_3.txt')
    write_to_file(action_four_behavior, dir_path+'action_4.txt')


def split_five_days_3_behavior():
    """
    获取4天，用户加入购物车的商品，排除已经购买的
    :return:
    """
    start_time = '2017-8-21 00:00:00'
    end_time = '2017-8-25 23:59:59'

    start_time = date_to_timestamp(start_time)
    end_time = date_to_timestamp(end_time)

    behaviors = process_user_behaviors()
    seven_days = filter_time(behaviors, start_time, end_time)
    shopping_cart = filter_user_action(3, seven_days)
    purchased = filter_user_action(4, seven_days)
    print("start to remove")
    print(len(purchased), len(shopping_cart))
    _shopping = remove_purchased(purchased, shopping_cart)

    result = [(x[0], x[1]) for x in _shopping]
    output.output_result(result)


# 判断商品取消加入购物车

def not_buying():

    """
    下次付款前未购买的商品
    1. 加入购物车
    2. 浏览的商品
    3. 收藏的商品
    :return:
    """

    purchased_ = process_user_behaviors('data/behaviors/action_4.txt')

    purchased_dict = {}

    for pur_ in purchased_:
        if pur_[0] not in purchased_dict.keys():
            purchased_dict[pur_[0]] = dict()
        if pur_[1] not in purchased_dict[pur_[0]].keys():
            purchased_dict[pur_[0]][pur_[1]] = 0

        purchased_dict[pur_[0]][pur_[1]] += 1
    print("Load Purchased")
    del purchased_

    unpurchased_data = []

    def former_3_days_data(data):

        start_time = '2017-7-26 00:00:00'
        end_time = '2017-8-22 23:59:59'

        start_time = date_to_timestamp(start_time)
        end_time = date_to_timestamp(end_time)

        data = filter_time(data, start_time, end_time)

        return data

    action_1_data = process_user_behaviors('data/behaviors/action_1.txt')

    for dat in action_1_data:
        if dat[0] not in purchased_dict:
            unpurchased_data.append(dat)
        elif dat[1] not in purchased_dict[dat[0]]:
            unpurchased_data.append(dat)
    del action_1_data

    print("Finish Action 1: %s" % len(unpurchased_data))

    action_2_data = process_user_behaviors('data/behaviors/action_2.txt')

    for dat in action_2_data:
        if dat[0] not in purchased_dict:
            unpurchased_data.append(dat)

        elif dat[1] not in purchased_dict[dat[0]]:
            unpurchased_data.append(dat)

    del action_2_data

    print("Finish Action 2: %s" % len(unpurchased_data))

    action_3_data = process_user_behaviors('data/behaviors/action_3.txt')

    for dat in action_3_data:
        if dat[0] not in purchased_dict:
            unpurchased_data.append(dat)
        elif dat[1] not in purchased_dict[dat[0]]:
            unpurchased_data.append(dat)
    del action_3_data

    print("Finish Action 3: %s" % len(unpurchased_data))

    unpurchased_data = former_3_days_data(unpurchased_data)

    print("Not Buying: %s" % len(unpurchased_data))

    write_to_file(unpurchased_data,'data/behaviors/notbuying.txt')


def collect_user_product(action):

    """
    根据用户进行商品分类
    :param action 1/2/3/4
    :return:
    """

    action_data = process_user_behaviors('data/behaviors/action_%s.txt' % action)

    user_product = {}

    for element in action_data:
        if element[0] not in user_product.keys():
            user_product[element[0]] = []
        user_product[element[0]].append(element)

    return user_product


def extract_features_from_merchandises(merchandise):
    """
    从商品中提取出人物特征
    :param merchandise:
    :return:
    """

    pass


if __name__ == '__main__':
    not_buying()









































