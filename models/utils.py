import math
import torch
import torch.nn as nn

from torch.autograd import Variable

from libsvm.svmutil import *
from torch.utils import data

def dataset_init(dataset='a1a'):
    if dataset == 'a1a':
        train_dir = 'dataset/a1a/train.txt'
        test_dir = 'dataset/a1a/test.txt'
    elif dataset == 'gisette':
        train_dir = 'dataset/gisette/gisette_scale'
        test_dir = 'dataset/gisette/gisette_scale.t'
    elif dataset == 'usps':
        train_dir = 'dataset/usps/usps'
        test_dir = 'dataset/usps/usps.t'
    else:
        print("Dataset was not found")
    return train_dir, test_dir


# 数据集路径
train_dir = 'dataset/gisette/gisette_scale'
test_dir = 'dataset/gisette/gisette_scale.t'
# train_dir = 'dataset/a1a/train.txt'
# test_dir = 'dataset/a1a/test.txt'
# 损失测试了交叉熵BCE和差方和MSE，基本无区别
criterion = nn.MSELoss()

def learning_rate(init, epoch):
    # 随epoch增加到一定程度降低lr
    optim_factor = 0
    if(epoch > 1000):
        optim_factor = 3
    elif(epoch > 600):
        optim_factor = 2
    elif(epoch > 200):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_data(dataset='a1a'):
    train_dir, test_dir = dataset_init(dataset)
    # 因为数据集是从LibSVM拿的直接用它的读取程序
    y, x = svm_read_problem(train_dir, return_scipy=True)
    yt, xt = svm_read_problem(test_dir, return_scipy=True)
    traindata = (y, x)
    testdata = (yt, xt)
    return traindata, testdata


# 如果要用pytorch读取大型data方便分批次等操作的话，补全这个类，并使用Dataloader
# 参照 https://blog.csdn.net/guyuealian/article/details/88343924
# 这里数据量较低，暂时不用
class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        #
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        #
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# 单个epoch下train
def train(model, y, x, epoch, epoch_n, mini_batch, lr=0.001):
    model.train()
    lr = learning_rate(lr, epoch)
    # 简单归一化，换数据集时别忘记修正
    y = (y + 1) / 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 这里本该随着Dataloader有个mini-batch的，但是由于数据量小,先用这种方式代替
    batch_idx = x.shape[0] // mini_batch + 1
    acc_b = 0
    for batch in range(1, batch_idx+1):
        if batch == batch_idx:
            x_b = x[(batch-1) * mini_batch:, :, :]
            y_b = y[(batch-1) * mini_batch:, :]
        else:
            x_b = x[(batch-1) * mini_batch:batch * mini_batch, :, :]
            y_b = y[(batch-1) * mini_batch:batch * mini_batch, :]
        y_b, x_b = y_b.cuda(), x_b.cuda()
        y_b, x_b = Variable(y_b), Variable(x_b)
        optimizer.zero_grad()
        y_b_pred = model(x_b)
        loss = criterion(y_b_pred, y_b)
        loss.backward()
        optimizer.step()
        acc_b += torch.floor(y_b_pred / 0.5).eq(y_b).cpu().sum() / y_b.shape[0]
        max_p = max(y_b_pred)
        min_p = min(y_b_pred)

    print('用于检验训练错误，输出pred的最大值:%5f与最小值%5f' % (max_p, min_p))
    acc = acc_b / batch_idx
    return loss.data, acc, lr


def test(model, yt, xt):
    model.eval()
    yt = (yt + 1) / 2

    yt, xt = yt.cuda(), xt.cuda()
    yt, xt = Variable(yt), Variable(xt)
    yt_pred = model(xt)
    loss_t = criterion(yt_pred, yt)
    acc_t = torch.floor(yt_pred / 0.5).eq(yt).cpu().sum() / yt.shape[0]

    return loss_t.data, acc_t
