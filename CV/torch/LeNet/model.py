# -*- coding:utf-8 -*-
# @Time:2021/4/7 19:22
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: model.py
# software: PyCharm

import torch.nn as nn
import torch.nn.functional as F


# Pytorch Tensor 的通道排序：[batch, channel, height, width]

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # (input_channel, output_channel=kernel_num, kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)    # (kernel_size=2, stride=2)
        # ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # input(batch_size, 3, 32, 32) output(batch_size, 16, 28, 28)
        x = self.pool1(x)           # input(batch_size, 16, 28, 28) output(batch_size, 16, 14, 14)
        x = F.relu(self.conv2(x))   # input(batch_size, 16, 28, 28) output(batch_size, 32, 10, 10)
        x = self.pool2(x)           # input(batch_size, 32, 10, 10) output(batch_size, 32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # input(batch_size, 32, 10, 10) output(batch_size, 32 * 5 * 5)
        x = F.relu(self.fc1(x))     # input(batch_size, 32 * 5 * 5) output(batch_size, 120)
        x = F.relu(self.fc2(x))     # input(batch_size, 120)        output(batch_size, 84)
        x = self.fc3(x)             # input(batch_size, 84)        output(batch_size, 10)

        return x


# import torch
#
# input1 = torch.rand([32, 3, 32, 32])
# model = LeNet()
# print(model)
# print(model(input1))
