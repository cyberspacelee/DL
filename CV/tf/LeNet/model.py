# -*- coding:utf-8 -*-
# @Time:2021/4/7 21:54
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: model.py
# software: PyCharm

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# tensorflow 的通道排列顺序：[batch, height, width, channel]
# 此处利用 tensorflow Model Subclassing API 搭建网络

class MyModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')  # (filters, kernel_size)
        # Padding 分为： VALID or SAME
        # VALID: N = (W - F + 1)/S [向上取整]  w: 输入图片大小，F: kernel_size
        # SAME: N = W / S [向上取整]

        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)       # input[batch_size, 28, 28, 1]      output[batch_size, 26, 26, 32]
        x = self.flatten(x)     # input[batch_size, 26, 26, 32]     output[batch_size, 26 * 26 * 32]
        x = self.d1(x)          # input[batch_size, 26 * 26 * 32]   output[batch_size, 128]

        return self.d2(x)       # input[batch_size, 128]            output[batch_size,10]