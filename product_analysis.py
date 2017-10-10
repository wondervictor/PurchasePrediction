# -*- coding: utf-8 -*-


def get_same_class_product(class_id, product_data):
    """
    获得同类商品
    :param class_id: 商品类别
    :param product_data: 商品数据
    :return:
    """
    def _filter_(x):
        if x[2] == class_id:
            return True
        else:
            return False

    class_data = filter(_filter_, product_data)

    return class_data



def construct_product_features():
    """
    构建商品特征：
    1. 价格
    2. 描述特征
    3. 店铺评价
    4. 商品是否浏览过、收藏过、加购过
    :return:
    """
