'''
这里主要用于存储稀疏性，低秩性计算相关函数
'''

import torch
import torch.nn as nn

import os
import sys
import math
import numpy as np
import time

# 变换x到卷积低秩的x
def PCA_to_ConvLowr(feature):
    # input格式为(N, 1, F)，是tensor，如果真的用这一版的话要注意修改，不要来回变换tensor和numpy
    feature = torch.squeeze(feature)
    Y = feature.t()
    Y = Y.cpu().detach().numpy()
    A = learn_A_by_PCA(Y)
    Ay = np.dot(A, Y)
    feature_Convlr = torch.from_numpy(Ay).float().t()
    feature_Convlr = feature_Convlr.view(feature_Convlr.shape[0], 1, feature_Convlr.shape[1])
    return feature_Convlr, A

# 训练集不应当参与构建矩阵A
def PCA_for_test(feature, A):
    feature = torch.squeeze(feature)
    Y = feature.t()
    Y = Y.cpu().detach().numpy()
    # A = learn_A_by_PCA(Y)
    Ay = np.dot(A, Y)
    feature_Convlr = torch.from_numpy(Ay).float().t()
    feature_Convlr = feature_Convlr.view(feature_Convlr.shape[0], 1, feature_Convlr.shape[1])
    return feature_Convlr


# 这里做一个函数从生成矩阵到A，也即LbCNNM文章中的算法一
def learn_A_by_PCA(Y):
    # Y为F特征数*N样本数，也即是原本的样本
    # numpy内置SVD产生的V无需转置，U*sigma*V = 原矩阵
    UY, sigma, VY = np.linalg.svd(Y)
    # 这里暂时引入一个输出sigma奇异值看看



    # c = U.dot(np.diag(sigma)).dot(V)
    # 这里修改到不补0的版本
    # zero_mm = np.zeros((UY.shape))
    # B = np.concatenate((UY.T, zero_mm))
    # Uf, Vf = DFT_UV(2 * UY.shape[0])
    B = UY.T
    Uf, Vf = DFT_UV(UY.shape[0])
    A = np.dot(Vf, Uf.T).dot(B)
    return A

# 定义DFT_UV函数，输出m维下UV，见第二节B，这个函数只运行一次就好，不要循环运行
def DFT_UV(m):
    dftmtx = np.fft.fft(np.eye(m))
    real = np.real(dftmtx)
    imag = np.imag(dftmtx)
    # 分别对实部虚部做SVD
    U1, sigma1, V1 = np.linalg.svd(real)
    V1 = V1.T
    # sigma1 = np.diag(sigma1)
    index1 = sigma1 > 0.1
    U1 = U1[:, index1]
    V1 = V1[:, index1]

    U2, sigma2, V2 = np.linalg.svd(imag)
    V2 = V2.T
    # sigma2 = np.diag(sigma2)
    index2 = sigma2 > 0.1
    U2 = U2[:, index2]
    V2 = V2[:, index2]

    return np.concatenate((U1, U2), axis=1), np.concatenate((V1, V2), axis=1)

# 计算卷积矩阵秩的基尼系数，这里取前几项算平均
def compute_conv_gini(x):
    # x为输入，应为numpy(N,1,F)
    n = 10
    feature = torch.squeeze(x)
    feature = feature[:n].cpu().detach().numpy()
    gini = 0
    for i in range(n):
        conv = conv_matrix(feature[i])
        _, e, _ = np.linalg.svd(conv)
        gini += compute_gini(e)
    gini_avg = gini / n

    return gini_avg


# 定义基尼系数的计算方式
# 论文里确定矩阵的低秩性用的是奇异值的GINI，这里后面可以详细看看，说不定要改
def compute_gini(feature):
    # 输入feature是提取到的特征，numpy型数据，转成向量，获得长度，取绝对值变为标准的可measure非负序列
    feature = feature.reshape(-1)
    n = feature.shape[0]

    # 利用compare measure那篇文章的公式，求1范数，排序，求解gini
    feature = np.abs(feature)
    f_sum = np.sum(feature)
    feature = np.sort(feature)
    k = np.arange(n) + 1.
    gini = 1. - 2. * np.sum((n + 0.5 - k) / (f_sum * n) * feature)
    return gini

# 一维circshift
def circshift(x, k):
    # 目前先做一维的, X是原输入, np数组, k是移动步长, k为正时行向量循环右移
    x = x.reshape(-1)
    return np.concatenate((x[-k:], x[:-k]))

# 特征展开为向量得到卷积矩阵,k取-1则按卷积矩阵为方阵输出
def conv_matrix(feature,k=-1):
    # 输入feature应该是提取到的特征，numpy向量，获得长度，取绝对值变为标准的可measure非负序列
    feature = feature.reshape(-1)
    n = feature.shape[0]
    if k == -1:
        k = n
    conv_m = np.zeros((n, k))
    for i in range(k):
        conv_m[:, i] = circshift(feature, i)
    return conv_m


if __name__ == '__main__':
    # U, V = DFT_UV(5)
    test_gini = np.array([1, 2, 3, 4, 5])
    gini = compute_gini(test_gini)
    print(gini)

# 事实上好像不用搞那种生成矩阵
# def get_gematrix(x, m):
#     # 输入size为：特征数F*样本数N(相当于将一组特征视为原本的一个数的形式，将不同的样本视为类似序列的结构)
#     # 详细内容见ipad上笔记
#     len = length(x)
#     n_max = len - m + 1
#     G = zeros(m, n_max)
#     for i = 1 : n_max
#         G(:, i) = x(i : i + m - 1)
# return G



# 从输入y到Ay
# def feature_to_Af(feature):
#     conv_m = conv_matrix(feature)
#     A = get_A_with_PCA(conv_m)
#     Af = A.dot(feature.cpu().detach().numpy().reshape(-1))
#     return Af


# 修正后变换
# 用于已经足够稀疏的特征的转化（在这里弃用）
# def feature_to_Af(feature):
#     # 求V*U*feature 理想状态下转化为卷积低秩
#     feature = feature.cpu().detach().numpy().reshape(-1)
#     Uf, Vf = DFT_UV(feature.shape[0])
#     feature_lowrank = np.dot(Vf, Uf.T).dot(feature)
#     # 验证是否卷积低秩
#     conv_f_lowrank = conv_matrix(feature_lowrank)
#     _, sigma, _ = np.linalg.svd(conv_f_lowrank)
#     conv_fl_gini = compute_gini(torch.from_numpy(sigma))
#     return feature_lowrank, conv_fl_gini



# feature = torch.randn(2, 3, 2)
# conv_m = conv_matrix(feature,3)
# print(feature,conv_m)

# c = np.random.randn(3,3)
# A = get_A_with_PCA(c)

# feature = torch.randn(2, 3, 2)
# Af = feature_to_Af(feature)
# inp = compute_gini(feature)
# outp = compute_gini(torch.from_numpy(Af))
# print(Af)









