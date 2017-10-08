# -*- coding: utf-8 -*-

import time
import os
import sys
import re

import numpy as np


class User(object):

    def __init__(self, id, rank, gender, birth_date, age, baby_birth_date, baby_age=0, baby_gender=-99):
        self.id = id
        self.rank = rank
        self.gender = gender
        self.birth_date = birth_date
        self.age = age
        self.baby_birth_date = baby_birth_date
        self.baby_age = baby_age
        self.baby_gender = baby_gender


def date_timestamp(date):
    if len(date) == 0:
        return -1
    pattern = '%Y-%m-%d %H:%M:%S'
    strptime = time.strptime(date, pattern)
    timestamp = time.mktime(strptime)
    return timestamp


# Split line by '\t'
def split_t(line):
    line_elements = line.rstrip('\n\r').split('\t')
    return line_elements


def process_user_info():

    data_path = 'data/user_info.txt'

    with open(data_path, 'r') as f:
        user_lines = f.readlines()

    user_elements = [split_t(x) for x in user_lines]

    users = []

    for user_info in user_elements:

        user_id = int(user_info[0])
        user_rank = int(user_info[1])
        user_gender = int(user_info[2])
        user_birth = date_timestamp(user_info[3])
        user_age = int(user_info[4])
        user_baby_birth = date_timestamp(user_info[5])
        user_baby_age = int(user_info[6])
        user_baby_gender = int(user_info[7])

        user = User(
            user_id,
            user_rank,
            user_gender,
            user_birth,
            user_age,
            user_baby_birth,
            user_baby_age,
            user_baby_gender
        )

        users.append(user)

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
                '(', ')', '<', '>', '=', '|', 'a', '《', '》', '。']

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
    print(len(word_bag))
    dictionary = build_dict(word_bag, 'data/word.dict')

    output_file = open('data/products.txt', 'w+')

    for product in products:

        title = product[4]

        title = [dictionary.get(x, UNK_ID) for x in title]
        product[4] = title
        line = ','.join(['%s' % x for x in product[:4]])
        line += ','
        line += '%s,' % product[5]
        line += ':'.join(['%s' % x for x in title])
        line += '\n'

        output_file.write(line)

    output_file.close()

    return products









