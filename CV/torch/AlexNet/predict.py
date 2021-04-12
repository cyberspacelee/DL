# -*- coding:utf-8 -*-
# @Time:2021/4/11 17:21
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: predict.py
# software: PyCharm

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    img_path = './data/tulip.jpg'
    assert os.path.exists(img_path), '{} path not exist.'.format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    #  read class_indice
    json_path = './class_indices.json'
    assert os.path.exists(json_path), '{} path not exist.'.format(json_path)

    json_file = open(json_path, 'r')
    class_indice = json.load(json_file)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = './model/AlexNet.pth'
    assert os.path.exists(weights_path), '{} path not exist.'.format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict
        outputs = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(outputs, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # 最大值的索引转化成 numpy 格式，等价于 torch.argmax(predict).item()


    print_res = "class: {} prob: {:.3f}".format(class_indice[str(predict_cla)], predict[predict_cla].numpy())

    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    main()