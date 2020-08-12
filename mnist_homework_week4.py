#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from mnist import MNIST
import numpy as np
import cv2

def get_feature(x):
    # feature=[0,0,0,0]
    xa = np.array(x)
    x = xa.reshape(28,28)
    x = x.astype(np.uint8)
    winSize=(28,28)
    blockSize =(14,14)
    blockStride = (7,7)
    cellSize = (7, 7)
    nbins = 9
    winStride = (1, 1)
    padding = (0, 0)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    feature = hog.compute(x, winStride, padding)
    # 下面添加提取图像x的特征feature的代码
    # def get_shadow(x,dim):
    #     feature  =torch.sum(x,dim)
    #     feature = feature.float()
    #     ## 归一化
    #     for i in range(0,feature.shape[0]):
    #         feature[i]=feature[i]/sum(feature)
    #     feature = feature.view(1,28)
    #     return feature
    # feature  = get_shadow(xt,0)

    return torch.from_numpy(feature).reshape(1,-1)

def model(feature,weights0,weights1,weights2):
    y=-1
    # 下面添加对feature进行决策的代码，判定出feature 属于[0,1,2,3,...9]哪个类别
    feature = torch.cat((feature,torch.tensor(1.0).view(1,1)),1)
    h = feature.mm(weights0)
    h1 = torch.tanh(h).mm(weights1)
    h2 = torch.tanh(h1).mm(weights2)
    y = torch.sigmoid(h2)
    return y

def get_acc(image_data,image_label,weights0,weights1,weights2,start_i,end_i):
    correct=0
    for i in range(start_i,end_i):
        feature = get_feature(image_data[i])
        y = model(feature,weights0,weights1,weights2)
        gt = image_label[i]
        pred = torch.argmin(torch.abs(y-1)).item()
        if gt==pred:
            correct+=1
    return  float(correct/float(end_i-start_i))

def one_hot(gt):
    gt_vector = torch.zeros(1,10)
    # gt_vector *= 0.0
    gt_vector[0,gt] = 1.0
    return gt_vector

def train_model(image_data, image_label, weights0, weights1, weights2,lr):
    loss_value_before = 1000000000000000.
    loss_value = 10000000000000.
    for epoch in range(0, 3000):
        loss_value_before = loss_value
        loss_value = 0
        for i in range(0, 80):
            feature = get_feature(image_data[i])
            y = model(feature, weights0, weights1,weights2)
            gt = image_label[i]
            # 只关心一个值
            loss = torch.sum((y[0, gt:gt + 1] - gt).mul(y[0, gt:gt + 1] - gt))
            gt_vector = one_hot(gt)
            loss_value += loss.data.item()
            loss.backward()
            weights0.data.sub_(weights0.grad.data * lr)
            weights0.grad.data.zero_()
            weights1.data.sub_(weights1.grad.data * lr)
            weights1.grad.data.zero_()
            weights2.data.sub_(weights2.grad.data * lr)
            weights2.grad.data.zero_()
        train_acc = get_acc(image_data, image_label, weights0, weights1,weights2, 0, 80)
        test_acc = get_acc(image_data, image_label, weights0, weights1,weights2, 80, 100)
        print("epoch=%s,loss=%s/%s,train/test_acc:%s/%s" % (epoch, loss_value, loss_value_before, train_acc, test_acc))
    return weights0, weights1,weights2


if __name__ == "__main__":
    weights0 = torch.randn(325, 100, requires_grad=True)
    weights1 = torch.randn(100, 20, requires_grad=True)
    weights2 = torch.randn(20, 10, requires_grad=True)

    # 初始化，对结果影响非常大，未使用该初始化，准确率：20%，使用该初始化，准确率50%左右。
    torch.nn.init.kaiming_normal_(weights0)
    torch.nn.init.kaiming_normal_(weights1)
    torch.nn.init.kaiming_normal_(weights2)

    # minst 2828 dataset 60000 samples
    mndata = MNIST('./mnist/python-mnist/data/')
    image_data_all, image_label_all = mndata.load_training()
    image_data = image_data_all[0:100]
    image_label = image_label_all[0:100]
    lr = 0.005
    # 对模型进行训练：
    weights0, weights1,weights2 = train_model(image_data, image_label, weights0, weights1, weights2,lr)
    # 测试：
    correct = 0
    for i in range(80, 100):
        feature = get_feature(image_data[i])
        y = model(feature, weights0, weights1,weights2)
        gt = image_label[i]
        pred = torch.argmin(torch.abs(y - 1)).item()
        print("图像[%s]得分类结果是:[%s]" % (gt, pred))
        if gt == pred:
            correct += 1
    print("acc=%s" % (float(correct / 20.0)))
