# -*- coding:utf-8 -*-
# @Time:2021/4/7 22:11
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train.py
# software: PyCharm

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import MyModel


def main():
    mnist = tf.keras.datasets.mnist

    # download in the first time
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # add a channel dimension
    x_train = x_train[:, tf.newaxis]
    x_test = x_test[:, tf.newaxis]

    # create data generator
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model
    model = MyModel()

    # define loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # define optimizer
    optimizer = tf.keras.optimizers.Adam()

    # define train_loss and train_accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # define test_loss and test_accuracy
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # define test function including calculating loss and calculating accuracy
    @tf.function
    def test_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object((labels, predictions))

        gradient = tape.gradient((loss, model.trainable_variables))
        optimizer.apply_gradients(zip(gradient, model, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    





if __name__ == '__main__':
        main()