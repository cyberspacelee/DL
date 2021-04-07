# -*- coding:utf-8 -*-
# @Time:2021/4/7 20:06
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train.py
# software: PyCharm

import torch
import torchvision
import torch.nn as nn
from model import LeNet
from torch import optim
import matplotlib.pyplot as plt
import numpy as np


def main():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         # Convert a PIL Image or numpy.ndarray to tensor
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         # ((mean), (std)) 图像标准化 output = (input - 0.5)/0.5
         ]
    )

    # 50000 张图片，第一次使用，需将 download 设置成 True
    train_set = torchvision.datasets.CIFAR10(root='./data/',
                                             download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    #  10000 张验证图片
    val_set = torchvision.datasets.CIFAR10(root='./data/',
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # def IMG_show(img):
    #     img  = img /2 + 0.5
    #     np_img = img.numpy()
    #     plt.imshow(np.transpose(np_img, (1, 2, 0)))
    #     plt.show()
    #
    # print(','.join('%5s' % classes[val_label[i]] for i in range(4)))
    # IMG_show(torchvision.utils.make_grid(val_image))

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()  # 这里已包含 SoftMax，故在网路中不用加入
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        # 记录损失
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()
            # 清零历史梯度
            # 如果不清除历史梯度，就会对计算的历史梯度进行累加（通过这个特性能够变相实现一个很大 batch 数值的训练）

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()  # 参数更新

            running_loss += loss.item()
            if step % 500 == 499:  # 每隔 500 个 mini_batch，验证集验证
                with torch.no_grad():
                    # 验证集验证十不用记录梯度信息，减少内存占用
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy =  torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './model/Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
