# -*- coding: utf-8 -*-

"""
可视化
"""
import matplotlib.pyplot as plt

from data_process import process_user_behaviors, process_product_info


def get_price():
    """
    获取商品价格词典
    :return: dict {key: 商品ID, value: [商家ID, 价格]}
    """
    merchandise = process_product_info()

    price = {}

    for element in merchandise:
        price[element[0]] = [element[1], element[4]]

    del merchandise

    print("Loading Price Finished")
    return price


def generate_purchased_counter():
    """
    生成已经购买的所有商品的数量统计
    :return:
    """

    data_path = 'data/behaviors/action_4.txt'

    action_data = process_user_behaviors(data_path)

    merchandise_nums = {}

    for dat in action_data:
        if dat[1] not in merchandise_nums.keys():
            merchandise_nums[dat[1]] = 0
        merchandise_nums[dat[1]] += 1

    print("Generating Counter Finished")
    return merchandise_nums


def generate_price_counter():

    """
    生成价格数量统计
    :return:
    """
    price_dict = get_price()

    merchandise_counter = generate_purchased_counter()

    prices = {}

    for key in merchandise_counter.keys():
        price = price_dict[key][1]
        if price not in prices:
            prices[price] = 0
        prices[price] += merchandise_counter[key]
    del merchandise_counter
    del price_dict
    return prices


def draw_price_counter():
    """
    绘图：数量versus价格
    :return:
    """
    prices = generate_price_counter()

    # flatten to list

    print(len(prices))


def draw_merchandise_counter():

    """
    绘图: 商品数量与种类
    :return:
    """


if __name__ == '__main__':

    draw_price_counter()