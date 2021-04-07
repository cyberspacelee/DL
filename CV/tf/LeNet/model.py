# -*- coding:utf-8 -*-
# @Time:2021/4/7 21:54
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: model.py
# software: PyCharm

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)