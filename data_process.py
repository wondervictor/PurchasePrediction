# -*- coding: utf-8 -*-

import time
import os
import sys
import re

import numpy as np


class User(object):

    def __init__(self, id, rank, gender, has_baby, baby_age=0, baby_gender=-99):
        self.id = id
        self.rank = rank
        self.gender = gender
        self.has_baby = has_baby
        self.baby_age = baby_age
        self.baby_gender = baby_gender

    def __str__(self):
        return "[id]: %s [rank]: %s [gender]: %s [has_baby]: %s [baby_birth]: %s [baby_age]: %s [baby_gender]: %s" % \
               (self.id, self.rank, self.gender, self.has_baby, self.baby_birth_date, self.baby_age, self.baby_gender)


def date_timestamp(date):
    if len(date) == 0 or date == '-99':
        return -1
    pattern = '%Y-%m-%d %H:%M:%S'
    strptime = time.strptime(date, pattern)
    timestamp = time.mktime(strptime)
    return timestamp


# Split line by '\t'
def split_t(line):
    line_elements = line.rstrip('\n\r').split('\t')
    return line_elements


def preprocess_user_info():

    """
    预处理用户信息，删除无效信息:出生年月、年龄
    :return: [ID, Rank, Gender, Baby_Birth, Baby_Age, Baby_Gender]
    """

    data_path = 'data/user_info.txt'

    with open(data_path, 'r') as f:
        user_lines = f.readlines()

    def map_func(x):
        x = split_t(x)
        return x[:3] + x[6:]

    user_elements = [map_func(x) for x in user_lines]

    with open('data/user_info_clear.txt', 'w+') as f:
        for element in user_elements:
            line = '\t'.join(element)
            line += '\n'
            f.write(line)
    print("Clear User Info Data")


def process_user_info():

    data_path = 'data/user_info_clear.txt'

    with open(data_path, 'r') as f:
        user_lines = f.readlines()

    user_elements = [split_t(x) for x in user_lines]

    users = []

    for user_info in user_elements:

        user_id = int(user_info[0])
        user_rank = int(user_info[1])
        user_gender = int(user_info[2])

        user_baby_age = int(user_info[3])
        user_baby_gender = int(user_info[4])

        has_baby = True
        if user_baby_gender == -99:
            has_baby = False
        user = User(
            id=user_id,
            rank=user_rank,
            gender=user_gender,
            has_baby=has_baby,
            baby_age=user_baby_age,
            baby_gender=user_baby_gender
        )
        users.append(user)
    print("Finished Process User Info")
    return users


# Process User Behaviors 4583371
def process_user_behaviors(data_path='data/behavior_info.txt'):

    with open(data_path, 'r') as f:
        behavior_lines = f.readlines()[:-1]

    behavior_elements = [map(int, split_t(x)) for x in behavior_lines]

    del behavior_lines

    return behavior_elements


UNK_ID = 0
__UNK__ = '<UNK>'
# UNK: ~,@,#,:,','


# 移除特殊字符
def clear_special_chars(word_set):

    specials = ['~', '@', ',', '【', '】', '#', '$', '%', '&', '', ' ', '!', '+', '-', '/', '*',
                ':', ';', '?', '{', '}', '¥', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '(', ')', '<', '>', '=', '|', 'a', '《', '》', '。', '的', '地', '得']

    for spec in specials:
        if spec in word_set:
            del word_set[spec]


# 移除描述中的数字
def remove_digits(in_str):
    pattern = '[0-9]\d*'
    result = re.compile(pattern).findall(in_str)
    if len(result):
        return False

    return True


def build_dict(word_set, dict_file, size=60000):

    word_list = [__UNK__]
    if not os.path.exists(dict_file):
        clear_special_chars(word_set)
        word_list += sorted(word_set, key=word_set.get, reverse=True)
        if len(word_list) > size:
            word_list = word_list[:size]

        with open(dict_file, 'w+') as f:
            content = '\n'.join(word_list)
            f.write(content)
    del word_set
    del word_list

    with open(dict_file, 'r') as f:
        lines = f.readlines()
        word_list = [token.rstrip('\n') for token in lines]
        del lines
        word_dict = dict([(x, y) for (y, x) in enumerate(word_list)])

    del word_list

    return word_dict


def process_raw_products_info1():
    """
    处理原始Product信息，将其信息提取成为以下格式
    商品ID,商家ID,品牌ID,类型ID,价格,描述xx xx xxx xxx
    xxx已使用词典进行数字化
    """

    data_path = 'data/product_info.txt'

    with open(data_path, 'r') as f:
        product_lines = f.readlines()

    products = []

    for line in product_lines:
        line = split_t(line)
        product = map(int, line[:4])
        product.append(int(line[-1]))
        product.append(line[4])
        products.append(product)

    print(len(products))

    output_file = open('data/products.txt', 'w+')

    for product in products:

        line = '\t'.join(['%s' % x for x in product[:5]])
        line += '\t'
        if products[5][0] == ' ':
            line += '%s' % product[5][1:]
        else:
            line += '%s' % product[5]

        line += '\n'
        output_file.write(line)
    print("Saved")
    output_file.close()


def process_raw_products_info():
    """
    处理原始Product信息，将其信息提取成为以下格式
    商品ID,商家ID,品牌ID,类型ID,价格,描述xx:xx:xxx:xxx
    xxx已使用词典进行数字化
    """
    word_bag = dict()

    data_path = 'data/product_info.txt'

    with open(data_path, 'r') as f:
        product_lines = f.readlines()

    products = []

    for line in product_lines:
        line = split_t(line)
        product = map(int, line[:4])
        title = line[4].split(' ')
        product.append(title)
        product.append(int(line[-1]))

        for t in title:
            if remove_digits(t):
                if t not in word_bag:
                    word_bag[t] = 0
                word_bag[t] += 1
        products.append(product)
    print(len(products))
    dictionary = build_dict(word_bag, 'data/word.dict')

    output_file = open('data/products.txt', 'w+')

    def filter_func(m):
        s = []

        for _element in m:
            _element = dictionary.get(_element, UNK_ID)
            if _element != 0:
                s.append(_element)

        return s

    for product in products:

        title = product[4]

        title = filter_func(title)
        product[4] = title
        line = ','.join(['%s' % x for x in product[:4]])
        line += ','
        line += '%s,' % product[5]
        line += ':'.join(['%s' % x for x in title])
        line += '\n'

        output_file.write(line)

    output_file.close()

    return products


def process_product_info():

    """
    处理商品信息：
    :return:商品ID,商家ID,品牌ID,类型ID,价格,描述[xx,xx,xxx,xxx]
    """
    data_path = 'data/products.txt'

    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.rstrip('\n').split(',')
            if len(elements[-1]):
                describe_words = map(int, elements[-1].split(':'))
            else:
                describe_words = []
            elements = map(int, elements[:-1])
            elements.append(describe_words)
            data.append(elements)

    return data


def remove_unactive(under_action=1):

    """
    删除不活跃商品, 删除未被浏览过的商品
    :param under_action:
    :return:
    """

    active_merchandise = {}

    action_1_mer = process_user_behaviors('data/behaviors/action_1.txt')

    for mer in action_1_mer:
        active_merchandise[mer[1]] = 1

    del action_1_mer

    action_2_mer = process_user_behaviors('data/behaviors/action_2.txt')

    for mer in action_2_mer:
        active_merchandise[mer[1]] = 2

    del action_2_mer

    action_3_mer = process_user_behaviors('data/behaviors/action_3.txt')

    for mer in action_3_mer:
        active_merchandise[mer[1]] = 3

    del action_3_mer

    action_4_mer = process_user_behaviors('data/behaviors/action_4.txt')

    for mer in action_4_mer:
        active_merchandise[mer[1]] = 4

    del action_4_mer

    print("Active: %s" % len(active_merchandise))

    all_merchandises = process_product_info()

    print("All: %s" % len(all_merchandises))

    clear_merchandise = []
    for i in range(len(all_merchandises)):
        element = all_merchandises[i]
        if element[0] in active_merchandise:
            clear_merchandise.append(element)

    all_merchandises = clear_merchandise

    print("All After Clear: %s" % (len(all_merchandises)))

    data_path = 'data/products.txt'

    f = open(data_path, 'w+')

    for data_ele in all_merchandises:
        line = ','.join(['%s' % x for x in data_ele[:5]])
        line += ','
        line += ':'.join(['%s' % x for x in data_ele[5]])
        line += '\n'

        f.write(line)

    f.close()

    print("Clear Finished")


if __name__ == '__main__':

    remove_unactive()





