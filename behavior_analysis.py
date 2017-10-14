# -*- coding: utf-8 -*-

import time
import datetime
import random
from data_process import process_user_behaviors
import output

"""
浏览记录：3986041
收藏记录：5128
加购记录：477142
购买记录：115055

分割训练集测试集依据:

"""


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
    start_time = '2017-8-20 12:00:00'
    end_time = '2017-8-25 23:59:59'

    start_time = date_to_timestamp(start_time)
    end_time = date_to_timestamp(end_time)

    behaviors = process_user_behaviors()
    seven_days = filter_time(behaviors, start_time, end_time)
    shopping_cart = filter_user_action(3, seven_days)
    favorite_data = filter_user_action(2, seven_days)
    shopping_cart = favorite_data + shopping_cart
    #purchased = filter_user_action(4, seven_days)
    # print("start to remove")
    # print(len(purchased), len(shopping_cart))
    # _shopping = remove_purchased(purchased, shopping_cart)

    result = [(x[0], x[1]) for x in shopping_cart]
    output.output_result1(result)


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

    print("Finish Action 1: %s" % len(unpurchased_data))

    action_2_data = process_user_behaviors('data/behaviors/action_2.txt')

    for dat in action_2_data:
        if dat[0] not in purchased_dict:
            unpurchased_data.append(dat)

        elif dat[1] not in purchased_dict[dat[0]]:
            unpurchased_data.append(dat)

    print("Finish Action 2: %s" % len(unpurchased_data))

    action_3_data = process_user_behaviors('data/behaviors/action_3.txt')

    for dat in action_3_data:
        if dat[0] not in purchased_dict:
            unpurchased_data.append(dat)
        elif dat[1] not in purchased_dict[dat[0]]:
            unpurchased_data.append(dat)

    print("Finish Action 3: %s" % len(unpurchased_data))

    unpurchased_data = former_3_days_data(unpurchased_data)

    print("Not Buying: %s" % len(unpurchased_data))

    write_to_file(unpurchased_data, 'data/behaviors/notbuying.txt')



def collect_user_product(action):

    """
    根据用户进行商品分类
    :param action 1/2/3/4
    :return:
    """
    print("start read action")
    action_data = process_user_behaviors('data/behaviors/action_%s.txt' % action)

    user_product = {}
    product_freq = {}

    for element in action_data:
        if element[0] not in user_product:
            user_product[element[0]] = []
        user_product[element[0]].append(element[1])
        try:
            product_freq[element[1]] += 1
        except:
            product_freq[element[1]] = 1

    print("read action_%s.txt" % action)
    return user_product, product_freq


# 构造训练集:
# 采样
def sample_dataset(sample_num, dataset):
    dataset_num = len(dataset)
    sample_index = random.sample(range(dataset_num), sample_num)
    dataset = [dataset[p] for p in sample_index]
    return dataset


def generate_dataset_old(testing_size):

    buying_data = process_user_behaviors('data/behaviors/action_4.txt')
    notbuying_data = process_user_behaviors('data/behaviors/notbuying.txt')

    pos_num = len(buying_data)

    notbuying_action_1 = filter_user_action(1, notbuying_data)
    notbuying_action_2 = filter_user_action(2, notbuying_data)
    notbuying_action_3 = filter_user_action(3, notbuying_data)

    act_1_num = len(notbuying_action_1)
    act_2_num = len(notbuying_action_2)
    act_3_num = len(notbuying_action_3)

    sample_1_num = pos_num * act_1_num / (act_1_num + act_2_num + act_3_num)
    sample_2_num = pos_num * act_2_num / (act_1_num + act_2_num + act_3_num)
    sample_3_num = pos_num * act_3_num / (act_1_num + act_2_num + act_3_num)

    notbuying_action_1 = sample_dataset(sample_1_num, notbuying_action_1)
    notbuying_action_2 = sample_dataset(sample_2_num, notbuying_action_2)
    notbuying_action_3 = sample_dataset(sample_3_num, notbuying_action_3)

    notbuying_data = notbuying_action_1 + notbuying_action_2 + notbuying_action_3

    def add_label(label):
        def func(x):
            x.append(label)
            return x
        return func

    pos_add_label = add_label(1)
    neg_add_label = add_label(0)
    buying_data = map(pos_add_label, buying_data) #[x.append(1) for x in buying_data]
    notbuying_data = map(neg_add_label, notbuying_data) #[x.append(0) for x in notbuying_data]
    # shuffle

    random.shuffle(buying_data)
    random.shuffle(notbuying_data)

    train_data_pos = buying_data[:-testing_size/2]
    train_data_neg = notbuying_data[:-testing_size/2]

    test_data_pos = buying_data[-testing_size/2:]
    test_data_neg = notbuying_data[-testing_size/2:]

    train_data = train_data_neg + train_data_pos
    test_data = test_data_neg + test_data_pos
    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data


def generate_dataset(testing_size=20000):

    buying_data = process_user_behaviors('data/behaviors/action_4.txt')
    notbuying_data = process_user_behaviors('data/behaviors/notbuying.txt')

    start_date = "2017-7-26 00:00:00"
    end_date = "2017-8-10 23:59:59"

    start_date = date_to_timestamp(start_date)
    end_date = date_to_timestamp(end_date)

    train_buying_data = filter_time(buying_data, start_date, end_date)

    train_nobuying_data = filter_time(notbuying_data, start_date, end_date)

    predict_start_date = "2017-8-11 0:0:0"
    predict_end_date = "2017-8-15 23:59:59"

    predict_end_date = date_to_timestamp(predict_end_date)
    predict_start_date = date_to_timestamp(predict_start_date)

    test_predict_recommendation_data = filter_time(buying_data, predict_start_date, predict_end_date) + \
                                  filter_time(notbuying_data, predict_start_date, predict_end_date)

    predict_start_date = "2017-8-16 0:0:0"
    predict_end_date = "2017-8-18 23:59:59"

    predict_end_date = date_to_timestamp(predict_end_date)
    predict_start_date = date_to_timestamp(predict_start_date)

    test_predict_data = filter_time(buying_data, predict_start_date, predict_end_date)

    start_date = "2017-8-19 00:00:00"
    end_date = "2017-8-25 23:59:59"

    start_date = date_to_timestamp(start_date)
    end_date = date_to_timestamp(end_date)

    train_buying_data += filter_time(buying_data, start_date, end_date)

    train_nobuying_data += filter_time(notbuying_data, start_date, end_date)

    def add_label(label):
        def func(x):
            x.append(label)
            return x

        return func

    pos_add_label = add_label(1)
    neg_add_label = add_label(0)
    train_buying_data = map(pos_add_label, train_buying_data)  # [x.append(1) for x in buying_data]
    train_nobuying_data = map(neg_add_label, train_nobuying_data)  # [x.append(0) for x in notbuying_data]
    # shuffle

    random.shuffle(train_buying_data)
    random.shuffle(train_nobuying_data)

    train_data = train_buying_data + train_nobuying_data

    random.shuffle(train_data)

    start_date = "2017-8-21 00:00:00"
    end_date = "2017-8-25 23:59:59"

    start_date = date_to_timestamp(start_date)
    end_date = date_to_timestamp(end_date)

    predict_recommentation_data = filter_time(buying_data+notbuying_data, start_date, end_date)

    return train_data, test_predict_data, test_predict_recommendation_data, predict_recommentation_data


if __name__ == '__main__':
    #
    train_data, test_predict_data, test_predict_recom_data, predict_recom_data = generate_dataset()
    print(len(train_data), len(test_predict_data), len(test_predict_recom_data), len(predict_recom_data))
    write_to_file(train_data, 'data/train_data.txt')
    write_to_file(test_predict_recom_data, 'data/test_predict_recom_data.txt')
    write_to_file(test_predict_data, 'data/test_predict_data.txt')
    write_to_file(predict_recom_data, 'data/predict_data.txt')

    #split_five_days_3_behavior()


