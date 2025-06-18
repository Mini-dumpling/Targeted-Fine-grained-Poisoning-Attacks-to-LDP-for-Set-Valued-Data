# -*- coding: utf-8 -*-
# @Time : 2023/12/28  15:14
# @Author : 王源源
# @Project ：Set-Value-LDP 
# @File : AOA-TFIPA-R.py
# @Software : PyCharm


import numpy as np
import random
from random import choice
from collections import Counter


def encode(domain, data_user):
    l = list()
    d = len(domain)
    for data in data_user:
        temp = [0] * d
        for x in data:
            inx = domain.index(x)
            temp[inx] = 1
        l.append(temp)
    return l


def perturb(data_encode, domain, epsilon, c):
    d = len(domain)

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    for data in data_encode:
        for i in range(d):
            if np.random.uniform() > p:
                if data[i] == 1:
                    data[i] = 0
                elif data[i] == 0:
                    data[i] = 1
    return data_encode


def estimate(data_perturb, domain, epsilon, c):
    array = np.sum(data_perturb, axis=0)
    count_dict = dict(zip(domain, array))

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    n = len(data_perturb)

    es_data = []

    for x in domain:
        x_count = count_dict.get(x, 0)
        rs = (x_count - n * q) / (n * (p - q))
        es_data.append(rs)

    return es_data


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度不相等")
    squared_errors = [(x - y) ** 2 for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse


def calculate_mae(A, B):
    if len(A) != len(B):
        return None  # 如果两个列表长度不同，则返回None
    n = len(A)
    sum_abs_diff = 0.0  # 初始化绝对值差的总和
    for i in range(n):
        sum_abs_diff += abs(A[i] - B[i])  # 计算绝对值差并累加到总和中
    mae = sum_abs_diff / n  # 计算平均绝对值差
    return mae


def normalization(frequency: list, m: int):
    frequency_min = min(frequency)
    f_sum = 0
    frequency_new = []
    for f in frequency:
        f_sum = f_sum + (f - frequency_min)
    for f in frequency:
        f =((f - frequency_min) / f_sum) * m
        frequency_new.append(f)
    return frequency_new


def frequency_true(domain: list, data_user: list):
    num = len(data_user)
    d = len(domain)
    result = []
    for i in range(d):
        count = 0
        for j in range(num):
            if domain[i] in data_user[j]:
                count += 1
        result.append(count/num)
    return result


if __name__ == '__main__':

    domain = list(range(0, 32))
    d = len(domain)
    n = 1000
    ep = 1.0
    c = 4       # 每个用户数据长度

    times = 100

    for i in range(times):
        data_raw = generate_data(domain, n, c)  # 产生用户数据
        # print("用户数据data_raw：", data_raw)

        # 攻击前估计频率
        data_encode = encode(domain, data_raw)
        data_per = perturb(data_encode, domain, ep, c)
        data_es = estimate(data_per, domain, ep, c)
        # print(data_es)
        # print(len(data_es))
        # print(sum(data_es))

        data_es_nor = normalization(data_es, c)
        # print(data_es_nor)
        # print(len(data_es_nor))
        # print(sum(data_es_nor))

        data_es_true = frequency_true(domain, data_raw)
        # print(data_es_true)
        # print(len(data_es_true))
        # print(sum(data_es_true))

        print("MSE:", calculate_mse(data_es_nor, data_es_true))
        print("MAE:", calculate_mae(data_es_nor, data_es_true))

        print("--------------------------------------------------------")







