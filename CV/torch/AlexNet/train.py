# -*- coding:utf-8 -*-
# @Time:2021/4/11 15:13
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: train.py
# software: PyCharm

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {} device" .format(device))

    data_transform = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(224),  # 随机裁剪
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    }

    img_path = os.path.join(os.getcwd(), './data/')
    assert os.path.exists(img_path), '{} path not exist.'.format(img_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(img_path, 'train'),
                                         transform=data_transform['train'])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    # { "0": "daisy", "1": "dandelion", "2": "roses", "3": "sunflowers", "4": "tulips"}
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(img_path, 'val'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = AlexNet(num_classes=5, init_weights=False)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './model/AlexNet.pth'
    best_acc = 0.650
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # 启用 drop-out
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        # validate
        net.eval()  # 关闭 drop-out
        acc = 0.0
        # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # item() 的作用是取出单元素张量的元素值并返回该值，保持该元素类型不变
                # 使用 item() 函数取出的元素值的精度更高，所以在求损失函数等时我们一般用 item()

        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f"%(epoch+1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)


    print('Finish Traing')


if __name__ == '__main__':
    main()



