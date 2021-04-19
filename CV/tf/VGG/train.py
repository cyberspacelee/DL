# -*- coding:utf-8 -*-
# @Time:2021/4/13 20:35
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train.py
# software: PyCharm


import matplotlib.pyplot as plt
from model import vgg
import tensorflow as tf
import json
import os
import time
import glob
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)
            exit(-1)

    img_path = os.path.join(os.getcwd(), 'data')
    train_dir = os.path.join(img_path, 'train')
    validation_dir = os.path.join(img_path, 'val')
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("model"):
        os.mkdirs("model")

    im_height = 224
    im_width = 224
    batch_size = 3
    epochs = 10

    # get class dict
    data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    class_dict = dict((value, index) for index, value in enumerate(data_class))

    # transform value and key of dict
    inverse_dict = dict((value, key) for key, value in class_dict.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    # 使用 indent=4 这个参数对 json 数据格式化输出
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # load train images list
    train_image_list = glob.glob(train_dir+"/*/*.jpg")
    """
    glob 模块的主要方法就是 glob，该方法返回所有匹配的文件路径列表 list
    该方法需要一个参数用来指定匹配的路径字符串（字符串可以为绝对路径也可以为相对路径）
    其返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
    """
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    assert train_num > 0, "cannot find any .jpg file in {}".format(train_dir)
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]

    # load validation images list
    val_image_list = glob.glob(validation_dir+"/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("Using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    def process_path(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        # tf.io.read_file() 函数用于读取文件，相当于 open() 函数
        image = tf.image.decode_jpeg(image)  # 图片解码
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])

        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    # tf.data.Dataset 多线程， ImageDataGenerator 无多线程
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
        .map(process_path, num_parallel_calls=AUTOTUNE)\
        .repeat().batch(batch_size).prefetch(AUTOTUNE)

    # load validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .repeat().batch(batch_size)

    # 实例化模型
    model_name = 'vgg11'
    model = vgg(model_name, im_height=im_height, im_width=im_width, num_classes=5)
    # model = AlexNet(num_classes=5)
    # model.build((batch_size, 224, 224, 3)) # when using subclass model
    model.summary()

    # # using keras low api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')
    #
    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # val_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='val_accuracy')
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    # @tf.function
    # def val_step(images, labels):
    #     predictions = model(images, training=False)
    #     v_loss = loss_object(labels, predictions)
    #
    #     val_loss(v_loss)
    #     val_accuracy(labels, predictions)
    #
    # best_val_loss = float('inf')
    # train_step_num = val_num // batch_size
    # val_step_num = val_num // batch_size
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    #     val_loss.reset_states()
    #     val_accuracy.reset_states()
    #
    #     t1 = time.perf_counter()
    #     for index, (images, labels) in enumerate(train_dataset):
    #         train_step(images, labels)
    #         if index + 1 == train_step_num:
    #             break
    #
    #     print(time.perf_counter()-t1)
    #
    #     t2 = time.perf_counter()
    #     for index, (images, labels) in enumerate(val_dataset):
    #         val_step(images, labels)
    #         if index + 1 == val_step_num:
    #             break
    #
    #     print(time.perf_counter() - t2)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, val Loss: {}, val Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           val_loss.result(),
    #                           val_accuracy.result() * 100))
    #     # if val_loss.result() < best_val_loss:
    #     #     model.save_weights("./model/AlexNet.ckpt", save_format='tf')
    #
    # model.save_weights("./model/AlexNet.h5")

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./model/VGG.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]
    #
    # # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_dataset,
                        steps_per_epoch=train_num // batch_size,
                        epochs=epochs,
                        validation_data=val_dataset,
                        validation_steps=val_num // batch_size,
                        callbacks=callbacks)


if __name__ == '__main__':
    main()
