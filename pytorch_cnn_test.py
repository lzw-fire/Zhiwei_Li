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

import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
if_use_gpu = 1

def decode_idx3_ubyte(idx3_ubyte_file):
    """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集标签
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

# 获取训练集dataset
train_data_file = 'D:/data/Mnist-Data-Set/train-images-idx3-ubyte'
train_lable_file = 'D:/data/Mnist-Data-Set/train-labels-idx1-ubyte'
test_data_file = 'D:/data/Mnist-Data-Set/t10k-images-idx3-ubyte'
test_lable_file = 'D:/data/Mnist-Data-Set/t10k-labels-idx1-ubyte'

train_data = decode_idx3_ubyte(train_data_file)
train_lable = decode_idx1_ubyte(train_lable_file)
test_data = decode_idx3_ubyte(test_data_file)
test_lable = decode_idx1_ubyte(test_lable_file)

# 打印MNIST数据集的训练集及测试集的尺寸
print(train_data.shape)
print(train_lable.shape)
print(test_data.shape)
print(test_lable.shape)
test_data = Variable(torch.unsqueeze(torch.Tensor(test_data), dim=1),
         volatile=True).type(torch.FloatTensor)[:10000]/255
test_lable = torch.Tensor(test_lable)
train_data = Variable(torch.unsqueeze(torch.Tensor(train_data), dim=1),
         volatile=True).type(torch.FloatTensor)[:20000]/255
train_set = TensorDataset(torch.Tensor(train_data[:20000]), torch.Tensor(train_lable[:20000]))
# test_set = TensorDataset(test_data, test_lable)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# plt.imshow(training_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % training_data.train_labels[0])
# plt.show()


# test_y = test_y.cuda()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (16,28,28)
            # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 将（batch，32,7,7）展平为（batch，32*7*7）
        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)
# x = torch.rand(1, 1, 28, 28)
# y = cnn(x)
# g = make_dot(y)
# g.render('espnet_model', view=False)
if if_use_gpu:
    cnn = cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    start = time.time()
    train_correct = 0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output = cnn(b_x)
        loss = loss_function(output, b_y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, train_predict = torch.max(output.data, 1)
        train_correct += torch.sum(train_predict == b_y)

    if if_use_gpu:
        test_data = test_data.cuda()
        test_lable = test_lable.cuda()
    test_output = cnn(test_data)
    __, pred_y = torch.max(test_output, 1)
    # accuracy = float((pred_y == test_lable[:1000].data.squeeze()).astype(int).sum()) / float(test_lable[:1000].size(0))
    accuracy = int(torch.sum(pred_y == test_lable[:10000])) / 10000

    train_acc = int(train_correct) / len(train_loader.dataset)
    print('Epoch:', epoch, '|Step:', step,'|train loss:%.4f' % loss.item(),'test_acc:', accuracy)
    print('ACC = ', train_acc)
    duration = time.time() - start
    print('Training duation: %.4f' % duration)

# cnn = cnn.cpu()
# test_output = cnn(test_data[:1000])
# pred_y = torch.max(test_output, 1)[1].data.squeeze()
# s = sum(pred_y == test_lable[:1000])
# q = test_lable[:1000].size(0)
# accuracy = sum(pred_y == test_lable[:1000]).item() / test_lable[:1000].size(0)
# print('Test Acc: %.4f' % accuracy)