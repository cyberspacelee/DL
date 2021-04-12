# -*- coding:utf-8 -*-
# @Time:2021/4/11 14:49
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: model.py
# software: PyCharm

import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),      # input [batch_size, 3, 224, 224] output [batch_size, 48, 55, 55]
            # padding int or tuple
            # int: 上，下，左，右
            # tuple: 如：(1, 2) 1 代表上下方各补一列 0，2 代表左右两侧各补两列 0
            # nn.ZeroPad2d((1, 2, 1, 2)) 左：1 右：2 上：1 下：2
            # Pytorch  计算卷积或池化后的 size 时，如果有小数，会舍弃（如舍弃最右侧和最下侧的 0）
            nn.ReLU(inplace=True),
            # inplace 为 True，将会改变输入的数据（小于 0 的部分置为 0， 节省内存） ，否则不会改变原输入，只会产生新的输出
            nn.MaxPool2d(kernel_size=3, stride=2),                      # input [batch_size, 48, 55, 55] output [batch_size, 48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),               # input [batch_size, 48, 27, 27] output [batch_size, 128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # input [batch_size, 128, 27, 27] output [batch_size, 128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),              # input [batch_size, 128, 13, 13] output [batch_size, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),              # input [batch_size, 192, 13, 13] output [batch_size, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),              # input [batch_size, 192, 13, 13] output [batch_size, 128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                       # input [batch_size, 128, 13, 13] output [batch_size, 128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # [batch_size, C, H, W]
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # mean: 0  std^2: 0.01
                nn.init.constant_(m.bias, 0)
