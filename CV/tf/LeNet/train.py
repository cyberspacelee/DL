# -*- coding:utf-8 -*-
# @Time:2021/4/7 22:11
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train.py
# software: PyCharm

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import MyModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    mnist = tf.keras.datasets.mnist

    # download in the first time
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    imgs = x_train[:3]
    labs = y_train[:3]
    print(labs)
    plot_imgs = np.hstack(imgs)
    plt.imshow(plot_imgs, cmap='gray')
    plt.show()


    # add a channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

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
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # test function
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == '__main__':
    main()

    



