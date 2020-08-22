import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorwatch as tw
import torchvision.models
from torch.utils.data import TensorDataset, DataLoader
from torchviz import make_dot
import time
import struct
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 100
BATCH_SIZE = 20
LR = 0.0001
if_use_gpu = 1


# X = np.load('O-01-angle-X.npy')   # 数据集
# Y = np.load('O-01-angle-Y.npy')   # 标签

X = np.load('O-01-delay-X.npy')   # 数据集
Y = np.load('O-01-delay-Y.npy')   # 标签



def data_prepare(data, lable):

    X = data.astype('float32') / 255.        # minmax_normalized

    print(X.shape)
    print(lable.shape)


    # transfer
    subcarrier_num = 3
    lenth_sample = 800  # 750


    X = X.reshape(
        X.shape[0],
        lenth_sample,
        subcarrier_num,)  # (***, 750, 90)

    print(X.shape)

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, lable, test_size=0.33, random_state=42)  # 打乱分离

    train_X = torch.unsqueeze(torch.Tensor(train_X), 1)  # 维度加1  (***,1,750,90)
    train_set = TensorDataset(
        torch.Tensor(train_X),
        torch.Tensor(train_Y))  # 将训练数据的特征和标签组合
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)  # 随机读取小批量

    test_X = torch.unsqueeze(torch.Tensor(test_X), 1)
    test_set = TensorDataset(
        torch.Tensor(test_X),
        torch.Tensor(test_Y))  # 将训练数据的特征和标签组合
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=True)  # 随机读取小批量


    print(train_X.size())
    print(train_Y.shape)
    print(test_X.size())
    print(test_Y.shape)

    print('data prepare over!')

    return train_loader, test_loader

train_loader, test_loader = data_prepare(X, Y)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input_shape(1,750,90)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 3),
                      stride=1, padding=2),  # output_shape(16,750,90)
            # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            # nn.BatchNorm2d(),
            nn.ReLU(),  # activation
            # 在(2,2)空间里向下采样，Output shape(16,375,46)
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv2 = nn.Sequential(  # (16,14,14)
        #     nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  # (32,7,7)
        # )
        self.out = nn.Linear(16 * 400 * 2, 4)  # 全连接层  输出4类

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        x = x.view(-1, 16 * 400 * 2)  # 将（batch，16,375,46）展平为（batch，16*375*46）
        output = self.out(x)
        return output


cnn = CNN()

if if_use_gpu:
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    start = time.time()
    train_correct = 0
    test_correct = 0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = cnn(b_x)
        train_loss = loss_function(output, b_y.long())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        _, train_predict = torch.max(output.data, 1)
        train_correct += torch.sum(train_predict == b_y)

    for test_step, (x, y) in enumerate(test_loader):
        test_x = Variable(x, requires_grad=False)
        test_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        # cnn = cnn.cpu()
        test_output = cnn(test_x)
        test_loss = loss_function(test_output, test_y.long())
        __, pred_y = torch.max(test_output, 1)
        # test_acc = float((pred_y == test_Y.data.squeeze()).astype(int).sum()) / float(test_Y.size(0))
        test_correct += torch.sum(pred_y == test_y)

    train_acc = int(train_correct) / len(train_loader.dataset)
    test_acc = int(test_correct) / len(test_loader.dataset)
    duration = time.time() - start
    print(
        'Epoch:',
        epoch +1,
        '|train loss:%.4f' %
        train_loss.item(),
        '|train_acc:%.4f ' %
        train_acc,
        '|test loss:%.4f' %
        test_loss.item(),
        '|test_acc: %.4f' %
        test_acc,
        '|Training duation: %.4f' %
        duration)
    # print('ACC = ', train_acc)

    # print('Training duation: %.4f' % duration)
# cnn = cnn.cpu()
# test_output = cnn(test_X)
# pred_y = torch.max(test_output, 1)[1].data.squeeze()
# # s = sum(pred_y == test_Y)
# # q = test_lable[:1000].size(0)
# test_acc = sum(pred_y == test_Y).item() / test_Y.size(0)  # 另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：
# print('test_acc: %.4f' % test_acc)
import os
PATH = 'D:/data/1/model.pth'
if not os.path.exists(PATH): #判断是否存在
    os.makedirs(PATH) #不存在则创建
torch.save(cnn.state_dict(), "model")
