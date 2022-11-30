import numpy as np
import torch
import torch.nn.functional as F

from models.utils import get_data, train, test
from models.CNN_1D import CNN_1D
from models.PCA import PCA_to_ConvLowr, PCA_for_test, compute_conv_gini
def direct_CNN():
    # 这里有个问题，这种getdata的方式太慢了，后面还是给他改喽，数据量也不大，咋这么慢
    # 获取多种数据集'a1a' 'gisette' 'usps'
    traindata, testdata = get_data('usps')
    targets, input_x = traindata
    targets_t, input_x_t = testdata




    targets, input_x = torch.tensor(targets, dtype=torch.float32), torch.tensor(input_x.todense(), dtype=torch.float32)
    targets_t, input_x_t = torch.tensor(targets_t, dtype=torch.float32), torch.tensor(input_x_t.todense(), dtype=torch.float32)
    # 重新整理数据形式，这里后面改到一个函数里
    # X(N样本数, C通道数=1, F特征数)
    # 这里存在一个问题，就是LibSVM的读取和存储方式会导致x和xt特征数不同（x中只出现了前119维，但xt出现了123维）---已解决，更换数据集时注意
    feature_n = max(input_x.shape[-1], input_x_t.shape[-1])
    zeros_pad = torch.zeros((input_x.shape[0], feature_n - input_x.shape[-1]))
    input_x = torch.cat((input_x, zeros_pad), dim=1)
    input_x = input_x.view(input_x.shape[0], 1, feature_n)
    input_x_t = input_x_t.view(input_x_t.shape[0], 1, feature_n)
    # targets = targets.view(targets.shape[0], 1)
    # targets_t = targets_t.view(targets_t.shape[0], 1)

    # 对于多分类任务用one-hot编码
    # 取余的原因是原来编码是1~10，我不清楚10代表啥，这里当作0了
    targets_one_hot = F.one_hot(targets.to(torch.int64) % 10).to(torch.float32)
    targets_t_one_hot = F.one_hot(targets_t.to(torch.int64) % 10).to(torch.float32)

    # 这里变换原输入到卷积低秩的形式
    is_lowr = True
    is_gini = False
    if is_lowr:
        # 这里是分别做了生成矩阵和变换，联合起来做可能效果更好？感觉应该联合起来做
        # 应该先检查下是否变得低秩了
        if is_gini:
            gini_x = compute_conv_gini(input_x)
            gini_xt = compute_conv_gini(input_x_t)
        input_x, A = PCA_to_ConvLowr(input_x)
        input_x_t = PCA_for_test(input_x_t, A)
        if is_gini:
            gini_x_lr = compute_conv_gini(input_x)
            gini_xt_lr = compute_conv_gini(input_x_t)
            print('训练集变换前gini系数为：%.4f 变换后gini系数为：%.4f' % (gini_x, gini_x_lr))
            print('测试集(用训练集的变换矩阵)变换前gini系数为：%.4f 变换后gini系数为：%.4f' % (gini_xt, gini_xt_lr))
            return
        # A为列正交没问题
        # I = np.dot(A.T,A)


    # 数据量有点大，分mini-batch训练
    mini_batch = 1024


    model = CNN_1D(input_x.shape[-1], dropout=0.5)
    model.cuda()
    epochs = 1500
    loss_t, acc_t = 0, 0
    for epoch in range(1, 1 + epochs):


        loss, acc, lr = train(model, targets_one_hot, input_x, epoch, epochs, mini_batch, lr=0.0001)


        loss_t, acc_t = test(model, targets_t_one_hot, input_x_t)
        print('epoch: %3d  loss: %.4f  acc: %.3f%%  lr: %.6f  loss_t: %.4f  acc_t: %.3f%%'
              % (epoch, loss, acc*100, lr, loss_t, acc_t*100))




if __name__ == '__main__':
    direct_CNN()


