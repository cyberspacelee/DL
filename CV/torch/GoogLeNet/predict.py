# -*- coding:utf-8 -*-
# @Time:2021/4/14 20:51
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: predict.py
# software: PyCharm

import os
import json
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import GoogLeNet

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
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    # load model weights
    weights_path = './model/GoogLeNet.pth'
    assert os.path.exists(weights_path), "cannot find {}".format(weights_path)

    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)  # strict=False 不会严格匹配权重

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}  prob: {:.3}  ".format(class_indice[str(predict_cla)],
                                                  predict[predict_cla].numpy())

    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    main()