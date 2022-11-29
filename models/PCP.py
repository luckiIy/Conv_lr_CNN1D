import torch
import torch.nn as nn

import math
import numpy as np
import sys
import time

import torch
import math
import numpy as np
#gini系数，x为向量，tensor([length])
def Comp_Gini(x):
    length = x.size(0)
    x = torch.abs(x)
    x_l1 = torch.sum(x, dim=0)
    x_l1 = x_l1.item()
    x, _ = torch.sort(x, dim=0, descending=False)
    y = 0
    for k in range(0, length):
        y = y + x[k].item() * (length - k - 0.5) / (length*x_l1)
        print(y)
    return 1 - 2 * y
#生成卷积矩阵
def Conv_mat(vec_a,kernel):
    w = vec_a.size(0)
    A_k = vec_a.unsqueeze(-1)
    for i in range(1,kernel):
        a_w=torch.tensor([vec_a[w-1].item()])
        a_1=vec_a[0:w-1]
        vec_a=torch.cat((a_w,a_1),dim=0)
        T_a=vec_a.unsqueeze(-1)
        A_k = torch.cat((A_k,T_a),dim=1)
    return A_k

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

    return np.concatenate((U1, U2),axis=1), np.concatenate((V1, V2), axis=1)

# pcp主成分追踪，L是低秩但不稀疏，S稀疏但不低秩，通过ALM解min_{L,S} |L|_*+lambda|S|_1,s.t.,D=L+S
#输入为np矩阵
def Com_pcp(D,lambda_=0,display=False):
    tol =1e-7
    maxIter = 1000
    m= np.size(D,0)
    n=np.size(D,1)
    if lambda_==0:
        lambda_=1/ math.sqrt(max(m,n))
    #initialize
    Y = D
    norm_two=np.linalg.norm(D,2)
    norm_inf = np.linalg.norm(D, np.inf)
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    L=np.zeros((m,n))
    mu = 1.25/norm_two #可以调整
    mu_bar = mu*1e7
    rho = 1.4+0.1*np.tanh(40-min(m,n))#可以调整
    d_norm = np.linalg.norm(D,'fro')

    iter = 0
    while(iter<maxIter):
        iter =iter+1

        temp=D-L +(1/mu)*Y
        S=np.where(temp-lambda_/mu>0 , temp-lambda_/mu, 0)+np.where(temp + lambda_/mu<0 , temp + lambda_/mu, 0)


        temp = D - S + (1 / mu) * Y
        L = D_mu(temp,1/mu)

        Z=D-L-S
        Y= Y+ mu*Z
        mu=min(mu*rho,mu_bar)

        ##stop Criterion
        stopCriterion=np.linalg.norm(Z,'fro')/d_norm
        if display and(iter% 100 == 0 or iter ==1 or stopCriterion < tol):
            print("#iter"+str(iter)+",mu="+str(mu)+",stopALM(PCP)"+str(stopCriterion))
        if stopCriterion < tol:
            break

    return L,S

#L更新的D_mu函数
def D_mu(D,mu):
    U,sig,V = np.linalg.svd(D,full_matrices=0)
    sig=np.where(sig>mu,sig,0)#对奇异值进行筛选
    digS=np.diag(sig)
    L = np.dot(np.dot(U,digS),np.transpose(V))
    return L

#求解矩阵R min|RX-Y|_1,s.t R^TR=I_m
def solve_orth_admm(X,Y,display=False):
    maxIter = 2000
    iter = 0
    tol = 1e-7
    n=X.shape[1]
    p=Y.shape[0]
    fnorm = np.linalg.norm(Y,'fro')
    two_norm=np.linalg.norm(X,2)
    W = np.zeros(p, n) #Lagrangemultiplier
    L = np.zeros(p, n)
    mu =1/two_norm
    rho=1.1
    while iter<maxIter:
        iter =iter +1
        #update R
        temp=(Y+L-W/mu)*np.transpose(X)
        u,_,v=np.linalg.svd(temp)
        R=u*v
        #update L
        temp = R * X - Y + W / mu
        L = np.where(temp - 1/mu > 0, temp - 1/mu, 0) + np.where(temp + 1/mu < 0,temp + 1/mu, 0)

        H = R * X - Y - L

        #stop Criterion
        stopCriterion = np.linalg.norm(H, 'fro') / fnorm
        if display and(iter==1 or iter%100==0 or stopCriterion < tol):
            print("#iter" + str(iter) + ",mu=" + str(mu) + ",stopALM(PCP)" + str(stopCriterion))
        if stopCriterion < tol:
            break
        W = W + mu * H
        mu = min(mu * rho, 10 ^ 10)
    return R










# 用ADMM进行PCP分解，把输入矩阵D分解为稀疏和低秩的分量
# 这个暂时不能用，先搞PCA了
def PCP_withADMM(D, lambda_ = -1, display = False ):
    (m, n) = D.shape
    # 默认取lambda为1/√max(m, n)
    if lambda_ == -1:
        lambda_ = 1 / np.sqrt(max(m, n))
    # 定义最大迭代次数和迭代允许误差
    tol = 1E-7
    maxIter = 1000
    # 初始化,这里求输入二范数和无穷范数
    Y = D
    norm_two = np.linalg.norm(Y, 2)
    # 这里matlab版本的Y有个：，不明原因
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    # 下面是取了个最大的范数，然后做归一化？我寻思这也不是归一化啊
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm
    # 下面又是一些不明觉厉的初始化
    L = np.zeros(m, n)
    mu = 1.25/norm_two  # this one can be tuned
    mu_bar = mu * 1E+7
    rho = 1.4 + 0.1*np.tanh(40 - min(m,n))# this one can be tuned
    d_norm = np.linalg.norm(D, 'fro')

    iter = 0
    while iter < maxIter:
        iter += 1

        temp = D - L + (1/mu) * Y
        S = max(temp - lambda_/mu, 0) + min(temp + lambda_/mu, 0)
        temp = D - S + (1/mu)*Y

        [U, sig, V] = np.linalg.svd(temp, 'econ')
        # 这里有报错，先防止，先不实现PCP
        # svp = length(find(sig > 1/mu))
        sig = max(0,sig - 1/mu)
        svp = max(1, svp)




    return


