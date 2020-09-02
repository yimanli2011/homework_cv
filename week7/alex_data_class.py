#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch.nn as nn
from PIL import Image

def default_loader(path):
    path_one = os.path.join('./', path)
    return Image.open(path_one).convert('RGB')

class MyDataset(nn.Module):
    def __init__(self, data_dir, transform=None, target_transform=None, loader=default_loader,train=True):
        super(MyDataset, self).__init__()
        imgs = []
        if train:
            fileList = fileListTrain
            labeldict = labelDictTrain
        else:
            fileList = fileListval
            labeldict = labelDistval
        key = "Image filename"
        files = os.listdir(data_dir)  # 得到文件夹下所有文件名称
        for file in files:  # 遍历文件夹
            if file in fileList:
                label = labeldict.get(file)
                txts = os.listdir(os.path.join(data_dir, file))
                for txtfile in txts:
                    if not os.path.isdir(txtfile):  # 判断是否是文件夹,不是文件夹才打开
                        fh = open(os.path.join(data_dir,file,txtfile), 'r')
                        for line in fh.readlines():
                            if key in line: #读取到‘Image filename’所在行
                                line = line.strip('\n')
                                line = line.split(':')
                                imgpath = line[1].lstrip(' "').strip('"')
                                imgs.append((imgpath,label))
                                break
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# Alexnet
class AlexNet(nn.Module):  # 定义网络
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(3*227*227)
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding=0,ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size = 5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.conv3 = nn.Conv2d(256,384,kernel_size = 3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(4096, 4)

    # 网络前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# voc2005数据集（部分）
fileListTrain = ['Caltech_motorbikes_side',
                'TUGraz_bike',
                'TUGraz_cars',
                'TUGraz_person']
labelDictTrain = {'Caltech_motorbikes_side':0,'TUGraz_bike':1,'TUGraz_cars':2,'TUGraz_person':3}
fileListval = ['bike','car','motorbike','pedestrian']
labelDistval = {'motorbike':0,'bike':1,'car':2,'pedestrian':3}