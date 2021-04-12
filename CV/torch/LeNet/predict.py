# -*- coding:utf-8 -*-
# @Time:2021/4/7 21:39
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: predict.py
# software: PyCharm

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('./model/lenet.pth'))

    im = Image.open('./data/car.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 增加新维度 [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
        # or
        # predict = torch.softmax(outps, dim=1)

    print(classes[int(predict)])


if __name__ == '__main__':
    main()