# -*- coding:utf-8 -*-
# @Time:2021/4/16 19:28
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train_gpu.py
# software: PyCharm


import matplotlib.pyplot as plt
from  model import GoogLeNet
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
    batch_size = 32
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
    train_image_list = glob.glob(train_dir + "/*/*.jpg")
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
    val_image_list = glob.glob(validation_dir + "/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)
    assert val_num > 0, "cannot find any .jpg file in {}".format(validation_dir)
    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("Using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    def process_train_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = tf.image.random_flip_left_right(image)
        image = (image -  0.5) / 0.5

        return image, label


    def process_val_img(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        image = (image -  0.5) / 0.5

        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    # tf.data.Dataset 多线程， ImageDataGenerator 无多线程
    train_dataset = train_dataset.shuffle(buffer_size=train_num) \
        .map(process_train_img, num_parallel_calls=AUTOTUNE) \
        .repeat().batch(batch_size).prefetch(AUTOTUNE)

    # load validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_val_img,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .repeat().batch(batch_size)
    # tf.data 模块运行时，使用多线程进行数据通道处理，从而实现并行，这种操作几乎是透明的
    # 只需要添加 num_parallel_calls 参数到每一个 dataset.map() call 中
    # tf.data.experimental.AUTOTUNE，根据可用的 CPU 动态设置并行调用的数量

    # create model
    model = GoogLeNet(im_height, im_width, num_classes=class_num, aux_logits=True)
    model.summary()

    # using keras low api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # logits 表示网络的直接输出，没经过 sigmoid 或者 softmax 的概率化
    # from_logits=False 就表示把已经概率化了的输出，重新映射回原值
    optimizer = tf.optimizers.Adam(learning_rate=0.0003)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            aux1, aux2, output = model(images, training=False)
            loss1 = loss_object(labels, aux1)
            loss2 = loss_object(labels, aux2)
            loss3 = loss_object(labels, output)
            loss = loss1 * 0.3 + loss2 * 0.3 + loss3
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        _, _, output = model(images, training=False)
        t_loss = loss_object(labels, output)

        val_loss(t_loss)
        val_accuracy(labels, output)

    best_val_loss = float('inf')
    train_step_num = val_num // batch_size
    val_step_num = val_num // batch_size
    for epoch in range(1, epochs+1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        t1 = time.perf_counter()
        for index, (images, labels) in enumerate(train_dataset):
            train_step(images, labels)
            if index + 1 == train_step_num:
                break

        print(time.perf_counter()-t1)

        t2 = time.perf_counter()
        for index, (images, labels) in enumerate(val_dataset):
            val_step(images, labels)
            if index + 1 == val_step_num:
                break

        print(time.perf_counter() - t2)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, val Loss: {}, val Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result(),
                              val_loss.result(),
                              val_accuracy.result()))
        # if val_loss.result() < best_val_loss:
        #     model.save_weights("./model/AlexNet.ckpt", save_format='tf')

    model.save_weights("./model/GoogLeNet.ckpt", save_format='tf')





if __name__ == '__main__':
    main()
