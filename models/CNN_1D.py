import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_1D(nn.Module):
    def __init__(self, feature_n, class_c, in_ch=1, dropout=0.5):
        super().__init__()
        # 过拟合和了，现在加入了瓶颈层稍微好一些，最好再引入dropout
        # 引入dropout后在train上准确率惊人的提升到了98.5%，但是在测试集上却没有更好，过拟合是严重存在的（当然可能是由于训练集远小于测试集的必然结果）
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

        self.conv1d_1 = nn.Conv1d(in_ch, 4, kernel_size=51, padding=25, padding_mode='circular')
        self.BN1 = nn.BatchNorm1d(4, affine=True)
        self.conv1d_2 = nn.Conv1d(4, 2, kernel_size=51, padding=25, padding_mode='circular')
        self.BN2 = nn.BatchNorm1d(2, affine=True)
        self.conv1d_3 = nn.Conv1d(2, 4, kernel_size=51, padding=25, padding_mode='circular')
        self.BN3 = nn.BatchNorm1d(4, affine=True)
        self.conv1d_4 = nn.Conv1d(4, 1, kernel_size=51, padding=25, padding_mode='circular')
        self.BN4 = nn.BatchNorm1d(1, affine=True)
        self.fc1 = nn.Linear(1*feature_n, int(class_c))



    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.BN3(x)
        x = self.relu(x)
        x = self.conv1d_4(x)
        x = self.dropout(x)
        x = self.BN4(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


# 这里是将全连接层改为平均值或最值池化的非正常版本，权做实验了（我记得如果前面的网络够深的话这样是没问题的，但我这个小网络我不确定）
class CNN_without_FC(nn.Module):
    def __init__(self, feature_n, class_c, in_ch=1, dropout=0.5):
        super().__init__()
        # 过拟合和了，现在加入了瓶颈层稍微好一些，最好再引入dropout
        # 引入dropout后在train上准确率惊人的提升到了98.5%，但是在测试集上却没有更好，过拟合是严重存在的（当然可能是由于训练集远小于测试集的必然结果）
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

        self.conv1d_1 = nn.Conv1d(in_ch, 8, kernel_size=51, padding=25, padding_mode='circular')
        self.BN1 = nn.BatchNorm1d(8, affine=True)
        self.conv1d_2 = nn.Conv1d(8, 4, kernel_size=51, padding=25, padding_mode='circular')
        self.BN2 = nn.BatchNorm1d(4, affine=True)
        self.conv1d_3 = nn.Conv1d(4, 8, kernel_size=51, padding=25, padding_mode='circular')
        self.BN3 = nn.BatchNorm1d(8, affine=True)
        self.conv1d_4 = nn.Conv1d(8, 2, kernel_size=51, padding=25, padding_mode='circular')
        self.BN4 = nn.BatchNorm1d(2, affine=True)
        # 引入全局平均值池化
        self.gap = nn.AdaptiveAvgPool1d(int(class_c))



    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.BN3(x)
        x = self.relu(x)
        x = self.conv1d_4(x)
        x = self.dropout(x)
        x = self.BN4(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.gap(x)
        x = self.softmax(x)
        return x
if __name__ == '__main__':
    # target output size of 5
    m = nn.AdaptiveAvgPool1d(5)
    input = torch.randn(1, 64, 8)
    output = m(input)
    print(output)