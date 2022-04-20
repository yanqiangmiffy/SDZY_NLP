#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : humeng
# @Time    : 2021/1/19
from pandas import DataFrame
import pandas as pd
from rouge.rouge import Rouge

rouge = Rouge()


def evaluate(data: DataFrame):
    """

    :param data:  [refs, pres]
    :return:
    """
    # 加空格
    # data['refs'] = data.refs.apply(lambda x: ' '.join(x).lower())
    # data['pres'] = data.pres.apply(lambda x: ' '.join(x).lower())
    data_len = data.shape[0]
    rouge_1 = [0, 0, 0]  # f,p,r
    rouge_2 = [0, 0, 0]
    rouge_l = [0, 0, 0]
    scores = rouge.get_scores(hyps=data['pres'].tolist(), refs=data['refs'].tolist())
    for score in scores:
        # rouge_1
        rouge1 = score.get('rouge-1')
        rouge_1[0] += rouge1.get('f')
        rouge_1[1] += rouge1.get('p')
        rouge_1[2] += rouge1.get('r')
        # rouge_2
        rouge2 = score.get('rouge-2')
        rouge_2[0] += rouge2.get('f')
        rouge_2[1] += rouge2.get('p')
        rouge_2[2] += rouge2.get('r')

        # rouge_l
        rougel = score.get('rouge-l')
        rouge_l[0] += rougel.get('f')
        rouge_l[1] += rougel.get('p')
        rouge_l[2] += rougel.get('r')

    print('evaluate result:')
    print("rouge_1， f:%.2f, p:%.2f, r:%.2f" % (rouge_1[0] / data_len, rouge_1[1] / data_len, rouge_1[2] / data_len))
    print("rouge_2， f:%.2f, p:%.2f, r:%.2f" % (rouge_2[0] / data_len, rouge_2[1] / data_len, rouge_2[2] / data_len))
    print("rouge_L， f:%.2f, p:%.2f, r:%.2f" % (rouge_l[0] / data_len, rouge_l[1] / data_len, rouge_l[2] / data_len))


# data_res = pd.DataFrame(
#     [['中粮寒地东北大米', '福临门东北大米']],
#     columns=['raw_title', 'pre_title'])

data_res = pd.read_csv('/home/humeng/e/projects/hot_search/modules/pointer-generator-master/logs/decode/pre_res_model_17000_2021-01-27_172419.csv',
                       sep=',', names=['raw_title', 'refs', 'pres'])
data_res = data_res[['refs', 'pres']]
evaluate(data_res)
