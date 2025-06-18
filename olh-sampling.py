# -*- coding: utf-8 -*-
# @Time : 2023/12/28  15:20
# @Author : 王源源
# @Project ：Set-Value-LDP 
# @File : olh-sampling.py
# @Software : PyCharm


import numpy as np
import math
import xxhash
import random


def set_to_single(data_user, n, c):
    data_sample = [0 for col in range(n)]
    for i in range(n):
        data_sample[i] = data_user[i][np.random.randint(c)]
    return data_sample


def OLH_Perturb_sample(data_sample, n, epsilon):
    g = math.ceil(math.exp(epsilon) + 1)
    print("g:", g)
    p = 1 / 2
    data_perturb = [0 for col in range(n)]
    for i in range(n):
        data_perturb[i] = xxhash.xxh32(str(data_sample[i]), seed=i).intdigest() % g
        t = np.random.random()
        if t > p:
            temp = np.random.randint(g)  # 随机返回一个[0, g-1]的一个整数
            while temp == data_perturb[i]:
                temp = np.random.randint(g)
            data_perturb[i] = temp
    return data_perturb


# k代表数据取值范围，这里为D，就不需要k了
def OLH_Aggregate_sample(data_perturb, n, c, epsilon, domain):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    d = len(domain)
    Z = [0 for col in range(d)]
    for i in range(d):
        t_count = 0
        for j in range(n):
            temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
            if temp == data_perturb[j]:
                t_count += 1
        Z[i] = c * (t_count / n - q) / (p - q)
    return Z


def generate_data(domain: list, n: int, c: int):
    data_user = []      # 所有用户数据
    for i in range(n):
        x_raw = random.sample(domain, c)
        data_user.append(x_raw)
    return data_user


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


if __name__ == '__main__':

    domain = list(range(0, 32))
    d = len(domain)
    n = 1000
    ep = 1.0
    c = 4       # 每个用户数据长度

    times = 10

    for i in range(times):
        data_raw = generate_data(domain, n, c)  # 产生用户数据
        # print("用户数据data_raw：", data_raw)

        # 抽样
        data_sample = set_to_single(data_raw, n, c)

        # 攻击前估计频率
        data_per = OLH_Perturb_sample(data_sample, n, ep)
        data_es = OLH_Aggregate_sample(data_per, n, c, ep, domain)
        # print(data_es)
        # print(len(data_es))
        # print(sum(data_es))

        data_es_nor = normalization(data_es, c)
        print(data_es_nor)
        print(len(data_es_nor))
        print(sum(data_es_nor))

        data_es_true = frequency_true(domain, data_raw)
        print(data_es_true)
        print(len(data_es_true))
        print(sum(data_es_true))

        print("MSE:", calculate_mse(data_es_nor, data_es_true))
        print("MAE:", calculate_mae(data_es_nor, data_es_true))

        print("--------------------------------------------------------")



