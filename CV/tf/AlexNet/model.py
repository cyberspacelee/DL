# -*- coding:utf-8 -*-
# @Time:2021/4/11 19:52
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: model.py
# software: PyCharm

from tensorflow.keras import layers, models, Model, Sequential


def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow 中的 tensor 通道排序是 [N, H, W, C]
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # input [batch_size, 224, 224, 3]  output [batch_size, 227, 227, 3]
    # ((top_pad, bottom_pad), (left_pad, right_pad))
    # Conv2D, MaxPool, defult padding='valid'
    x = layers.Conv2D(48, kernel_size=11, strides=2)(x)                          # input [batch_size, 227, 227, 3]  output [batch_size, 55, 55, 48]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # input [batch_size, 55, 55, 48]   output [batch_size, 27, 27, 48]
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)  # input [batch_size, 27, 27, 48]   output [batch_size, 27, 27, 128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # input [batch_size, 27, 27, 128]  output [batch_size, 13, 13, 128]
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # input [batch_size, 13, 13, 128]  output [batch_size, 13, 13, 192]
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # input [batch_size, 13, 13, 192]  output [batch_size, 13, 13, 192]
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)  # input [batch_size, 13, 13, 192]  output [batch_size, 13, 13, 128]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # input [batch_size, 13, 13, 128]  output [batch_size, 6, 6, 128]

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(num_classes)(x)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)

    return model


class AlexNet(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
        ])

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(2048, activation='relu'),
            layers.Dense(num_classes),
            layers.Softmax(),
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)

        return x