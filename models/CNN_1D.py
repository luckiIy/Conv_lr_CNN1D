import torch.nn as nn




class CNN_1D(nn.Module):
    def __init__(self, feature_n, in_ch=1, dropout=0.5):
        super().__init__()
        # 过拟合和了，现在加入了瓶颈层稍微好一些，最好再引入dropout
        # 引入dropout后在train上准确率惊人的提升到了98.5%，但是在测试集上却没有更好，过拟合是严重存在的（当然可能是由于训练集远小于测试集的必然结果）
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        self.conv1d_1 = nn.Conv1d(in_ch, 4, kernel_size=5, padding=2, padding_mode='circular')
        self.BN1 = nn.BatchNorm1d(4, affine=True)
        self.conv1d_2 = nn.Conv1d(4, 2, kernel_size=5, padding=2, padding_mode='circular')
        self.BN2 = nn.BatchNorm1d(2, affine=True)
        self.conv1d_3 = nn.Conv1d(2, 4, kernel_size=5, padding=2, padding_mode='circular')
        self.BN3 = nn.BatchNorm1d(4, affine=True)
        self.fc1 = nn.Linear(4*feature_n, 50)
        self.fc2 = nn.Linear(50, 1)


    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.dropout(x)
        x = self.BN3(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x