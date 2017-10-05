# -*- coding: utf-8 -*-

import time
import os

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
def process_user_behaviors():

    data_path = 'data/behavior_info.txt'

    with open(data_path, 'r') as f:
        behavior_lines = f.readlines()

    behavior_elements = [map(int, split_t(x)) for x in behavior_lines]

    del behavior_lines

    return behavior_elements


def build_dict(word_set, dict_file):

    word_dict = dict()
    if not os.path.exists(dict_file):
        index = 0
        while len(word_set):
            word = word_set.pop()
            word_dict[word] = index
            index += 1
        with open(dict_file, 'w+') as f:
            for key in word_dict.keys():
                f.write('%s:%s\n' % (key, word_dict[key]))
    else:
        with open(dict_file, 'r') as f:
            lines = f.readlines()
            for line in lines:

                key, index = line.rstrip('\n').split(':')
                word_dict[key] = int(index)
    return word_dict


def process_products_info():

    word_bag = set()

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
            word_bag.add(t)

        products.append(product)

    dictionary = build_dict(word_bag, 'data/word.dict')

    for product in products:

        title = product[4]
        title = [dictionary[x] for x in title]
        product[4] = title

    return products








