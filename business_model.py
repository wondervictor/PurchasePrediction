# -*- coding: utf-8 =*-


"""
商家模型
"""

def create_business_features(products, bought, favorite):

    # bought = collect_user_product(4)
    # favorite = collect_user_product(2)
    # products = get_products_feature()
    
    shop_bought = {}
    brand_bought = {}
    shop_favorite = {}
    brand_favorite = {}
    # shop_fans = {}
    # brand_fans = {}
    # shop_customer = {}
    # brand_customer = {}
    
    shop_path = "data/shop.txt"
    brand_path = "data/brand.txt"

    # for user_id in bought.keys():
    #     product_id = bought[user_id]
    #     shop_id = products[product_id][0]#商家id
    #     brand_id = products[product_id][1]#品牌id
    #     if shop_id not in shop_customer.keys():
    #         shop_customer[shop_id] = 1
    #     else:
    #         shop_customer[shop_id] += 1
        

    for products_id in bought.values():
        for product_id in products_id:
            shop_id = products[product_id][0]#商家id
            brand_id = products[product_id][1]#品牌id
            if shop_id not in shop_bought.keys():
                shop_bought[shop_id] = 1
            else:
                shop_bought[shop_id] += 1
            if brand_id not in brand_bought.keys():
                brand_bought[brand_id] = 1
            else:
                brand_bought[brand_id] += 1
    for products_id in favorite.values():
        for product_id in products_id:
            shop_id = products[product_id][0]#商家id
            brand_id = products[product_id][1]#品牌id
            if shop_id not in shop_favorite.keys():
                shop_favorite[shop_id] = 1
            else:
                shop_favorite[shop_id] += 1
            if brand_id not in brand_favorite.keys():
                brand_favorite[brand_id] = 1
            else:
                brand_favorite[brand_id] += 1
            
    return shop_bought,shop_favorite,brand_bought,brand_favorite



    """
    定义商家模型
    :param purchase_records:
    :return:
    """
