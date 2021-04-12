# -*- coding:utf-8 -*-
# @Time:2021/4/11 21:43
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: predict.py
# software: PyCharm

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import AlexNet, AlexNet_v1


def main():
    im_height = 224
    im_width = 224

    # load image
    img_path = "./data/tulip.jpg"
    assert os.path.exists(img_path), "cannot find file {}".format(img_path)
    img = Image.open(img_path)

    # resize img
    img = img.resize((im_height, im_width))
    plt.imshow(img)

    # scaling pixel value to 0-1
    img = np.array(img) / 255

    # add the image to a batch
    img = (np.expand_dims(img, 0))

    # read class_indices
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "cannot find {}".format(json_path)

    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    # create model
    # model = AlexNet_v1(num_classes=5)
    weights_path = './model/myAlexNet.h5'
    assert os.path.exists(weights_path), "cannot find {}".format(weights_path)
    # model.load_weights(weights_path)

    model = AlexNet(num_classes=5)
    model.build((32, 224, 224, 3))
    model.load_weights(weights_path, by_name=True)
    # 从 HDF5 文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。
    # 如果想将权重载入不同的模型（有些层相同）中，则设置 by_name=True，只有名字匹配的层才会载入权重
    model.summary()


    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class {} prob:{:.3f}".format(class_indices[str(predict_class)], result[predict_class])

    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    main()

