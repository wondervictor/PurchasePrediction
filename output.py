# -*- coding: utf-8 -*-

"""
Output the result to result file

"""

import zipfile


def output_result(pred_user_behaviors):

    """
    :param pred_user_behaviors: [(UserID, ProductID),....]
    :return: None
    """

    with open('output/output.txt', 'w+') as f:

        for tup in pred_user_behaviors:

            f.write('%s\t%s\n'%(tup[0], tup[1]))


def output_result1(pred_user_behaviors):

    """
    :param pred_user_behaviors: [(UserID, ProductID),....]
    :return: None
    """

    with open('output/output1.txt', 'w+') as f:

        for tup in pred_user_behaviors:

            f.write('%s\t%s\n'%(tup[0], tup[1]))
