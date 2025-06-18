# -*- coding: utf-8 -*-
# @Time : 2023/9/6  15:44
# @Author : 王源源
# @Project ：Set-Value-LDP 
# @File : privset.py
# @Software : PyCharm


import numpy as np
import numpy.random as r
from scipy.special import comb
import random
import math


class PrivSet:
    name = 'PRIVSET'
    ep = 0.0    # privacy budget epsilon

    d = 0       # domain size + maximum subset size
    m = 0       # maximum subset size
    trate = 0.0     # hit rate when true
    frate = 0.0     # hit rate when false
    normalizer = 0.0    # normalizer for proportional probabilities

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.normalizer = normalizer
        # print(np.exp(self.ep/2), self.trate, self.frate)

    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d+m, 0.0)     # 0数组
        infos = [None] * (d+m)              # None数组
        for k in range(1, d):
            interCount = comb(d+m, k) - comb(d, k)
            nonInterCount = comb(d, k)
            normalizer = nonInterCount + interCount*np.exp(ep)
            trate = comb(d+m-1, k-1) * np.exp(ep)/normalizer
            frate = (comb(d-1, k-1)+(interCount * k - comb(d+m-1, k-1) * m) * np.exp(ep) / d) / normalizer
            errorbounds[k] = (trate * (1.0-trate) + (d+m-1) * frate * (1.0-frate)) / ((trate-frate) * (trate-frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d])+1
        return [bestk]+infos[bestk]

    def randomizer(self, secrets, domain):
        pub = np.zeros(self.d+self.m, dtype=int)    # 初始化零数组
        probs = np.full(self.k+1, 0.0)              # 初始化零数组
        for inter in range(0, self.k+1):
            probs[inter] = comb(self.m, inter) * comb(self.d, self.k-inter) / self.normalizer
        probs = probs * np.exp(self.ep)
        probs[0] = probs[0] / np.exp(self.ep)   # inter=0，probs[0]=comb(self.d, self.k) / self.normalizer

        for inter in range(1, self.k+1):
            probs[inter] += probs[inter-1]

        p = r.random(1)[0]                          # 取数组中第一个数据，即取一个随机数
        sinter = 0
        while probs[sinter] <= p:
            sinter += 1

        # 填充数据集
        domain_pad = domain + []
        for i in range(self.m):
            domain_pad.append(self.d + i)

        remain = list(set(domain_pad)-set(secrets))     # domain应该是填充域
        pubset = random.sample(secrets, sinter) + random.sample(remain, self.k-sinter)
        for i in range(0, self.d+self.m):
            if i in pubset:
                pub[i] = 1
        return pub


class PrivSet_SERVER:
    ep = 0.0    # privacy budget epsilon
    d = 0       # domain size + maximum subset size
    m = 0       # maximum subset size
    trate = 0.0     # hit rate when true
    frate = 0.0     # hit rate when false
    normalizer = 0.0    # normalizer for proportional probabilities

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.normalizer = normalizer
        self.trate = comb(self.d + self.m - 1, self.k - 1) * np.exp(self.ep) / normalizer
        self.frate = (comb(self.d - 1, self.k - 1) + (interCount * self.k - comb(self.d + self.m - 1, self.k - 1) * self.m) * np.exp(self.ep) / self.d) / normalizer
        # print(np.exp(self.ep/2), self.trate, self.frate)

    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d+m, 0.0)
        infos = [None] * (d+m)
        for k in range(1, d):
            interCount = comb(d+m, k) - comb(d, k)
            nonInterCount = comb(d, k)
            normalizer = nonInterCount + interCount*np.exp(ep)
            trate = comb(d+m-1, k-1) * np.exp(ep) / normalizer
            frate = (comb(d-1, k-1) + (interCount * k - comb(d+m-1, k-1) * m) * np.exp(ep) / d) / normalizer
            errorbounds[k] = (trate * (1.0-trate) + (d+m-1) * frate * (1.0-frate)) / ((trate-frate) * (trate-frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d]) + 1
        return [bestk]+infos[bestk]

    def decoder(self, m, domain, hits):
        # debias hits but without projecting to simplex
        es_data = []

        # 填充项
        d = len(domain)
        domain_x = domain.copy()
        for i in range(m):
            x = d + i
            domain_x.append(x)

        # 在获取扰动数据中元素频率时，一定要用字典，可以大大节省运行时间
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain_x, array))

        num = len(hits)

        for x in range(0, self.d + self.m):
            x_count = count_dict.get(x, 0)
            fs = (x_count - num * self.frate) / (num * (self.trate-self.frate))
            es_data.append(fs)

        return es_data


# 产生预处理后的用户数据
def generate_data(domain:list, m:int, n:int):
    data_user = []  # 原始用户数据

    # 随机数产生区间
    m_arr = []
    for i in range(m):
        m_arr.append(i + 1)

    # 填充项
    d = len(domain)
    domain_x = []
    for i in range(m):
        x = d + i
        domain_x.append(x)

    # 产生用户数据
    for i in range(n):
        num = random.sample(m_arr, 1)[0]
        x_raw = random.sample(domain, num)
        j = 0
        while num < m:
            x_raw.append(domain_x[j])
            j = j + 1
            num = num + 1
        data_user.append(x_raw)

    return data_user


def run(data: list, domain: list, m: int, ep, k):
    per_data = []   # 扰动数据
    d = len(domain)
    privset = PrivSet(d, m, ep, k)
    for x in data:
        x_per = privset.randomizer(x, domain)
        per_data.append(x_per.tolist())
    return per_data


def frequency_es(per_data:list, d:int, m:int, ep, k):
    privset_server = PrivSet_SERVER(d, m, ep, k)
    fs = privset_server.decoder(m, domain, per_data)
    return fs


def frequency_true(domain: list, data_binary: list, m: int):
    fre_data = []

    # 填充项
    d = len(domain)
    domain_x = domain.copy()
    for i in range(m):
        x = d + i
        domain_x.append(x)

    # 在获取扰动数据中元素频率时，一定要用字典，可以大大节省运行时间
    array = np.sum(data_binary, axis=0)
    count_dict = dict(zip(domain_x, array))

    num = len(data_binary)

    for x in domain_x:
        x_count = count_dict.get(x, 0)
        fs = (x_count / (num * m)) * m
        fre_data.append(fs)
    return fre_data


def binary(data: list, d: int, m: int):
    data_binary = []
    for x in data:
        temp = np.zeros(d+m, dtype=int)  # 初始化零数组
        for j in range(d+m):
            if j in x:
                temp[j] = 1
        data_binary.append(temp.tolist())
    return data_binary


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


# 求top-k频繁项， k=16
def top_k(k: int, data_p: dict):
    sorted_dict = dict(sorted(data_p.items(), key=lambda item: item[1], reverse=True))
    item = list(sorted_dict.keys())
    result = item[:k]
    return result


# F1指标
def F1_score(Lt: list, Lg: list):
    sett = set(Lt)
    setg = set(Lg)
    common = sett.intersection(setg)

    Lt_len = len(Lt)
    Lg_len = len(Lg)
    common_len = len(list(common))

    P = common_len / Lg_len
    R = common_len / Lt_len

    # F1 = 2*P*R / (P+R)
    F1 = P
    return F1


# NCR
def calculate_NCR(Lt: list, Lg: list, k: int):
    sum_Lt = (1 + k) * k / 2
    sum_Lg = 0
    for x in Lg:
        if x in Lt:
            r = Lt.index(x) + 1
            qv = k - (r - 1)
        else:
            qv = 0
        sum_Lg = sum_Lg + qv
    NCR = sum_Lg / sum_Lt
    return NCR


# MSE
def calculate_mse(list1: list, list2: list):
    if len(list1) != len(list2):
        return None

    squared_errors = [(x - y) ** 2 for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse


if __name__ == '__main__':

    domain = list(range(0, 32))
    d = len(domain)
    m = 4
    n = 1000
    k = None
    ep = 1.0

    # 产生用户数据，并做抽样或填充处理
    data_raw = generate_data(domain, m, n)
    # 处理成二进制数据
    data_binary = binary(data_raw, d, m)
    # 真实频率计算
    fre_true = frequency_true(domain, data_binary, m)
    print("真实频率：", fre_true)

    # 扰动
    per_data = run(data_raw, domain, m, ep, k)
    # 频率估计
    frequency = frequency_es(per_data, d, m, ep, k)
    print("扰动估计频率", frequency)

    # 计算MSE
    MSE = calculate_mse(fre_true, frequency)
    print("MSE:", MSE)

    # 填充数据集
    domain_pad = domain + []
    for i in range(m):
        domain_pad.append(d + i)
    print("domain_pad")
    print(domain_pad)

    k = 16
    raw_dict = dict(zip(domain_pad, fre_true))
    # 原始数据top-k列表
    At = top_k(k, raw_dict)
    print("原始数据top-k列表：", At)

    es_dict = dict(zip(domain_pad, frequency))
    # 估计数据top-k列表
    Ag = top_k(k, es_dict)
    print("估计数据top-k列表：", Ag)

    # 计算F1
    F1 = F1_score(At, Ag)
    print("F1:", F1)

    # 计算NCR
    NCR = calculate_NCR(At, Ag, k)
    print("NCR:", NCR)



