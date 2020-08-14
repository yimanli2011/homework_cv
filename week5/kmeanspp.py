#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from hct66 import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#必做：使用kmeans++ 对hct66数据集聚三类
image_data,image_label = generate_data()
data_feature = [np.array(get_feature(i)).reshape(6) for i in image_data]
model_1 = KMeans(n_clusters=3,init='k-means++')
model_1.fit_predict(data_feature)
label_pred_1 = model_1.labels_ #获取聚类标签
centroids_1 = model_1.cluster_centers_ #获取聚类中心
print(label_pred_1)
"""
#显示聚类结果
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
#这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
j=0
for i in label_pred:
    plt.plot([data_feature[j][1]], [data_feature[j][2]], mark[i], markersize = 5)
    j=j+1
plt.show()
"""
#可选：使用kmeans++ 对mnist数据集聚10类，查看聚类效果。
from mnist import MNIST
import cv2
# minst 2828 dataset 60000 samples
mndata = MNIST('../week4/mnist/python-mnist/data/')
image_data_all, image_label_all = mndata.load_training()
image_data_1000 = image_data_all[0:1000]
image_label_1000 = image_label_all[0:1000]
# 选用hog提取特征
def get_feature_hog(x):
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
    feature = torch.from_numpy(feature).reshape(2,-1)
    return feature.sum(dim=1)

data_feature = [np.array(get_feature_hog(i)).reshape(2) for i in image_data_1000]
model = KMeans(n_clusters=10,init='k-means++')
model.fit_predict(data_feature)
label_pred = model.labels_ #获取聚类标签
centroids = model.cluster_centers_ #获取聚类中心
print(label_pred)

#显示聚类结果
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
#这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
j=0
for i in label_pred:
    plt.plot([data_feature[j][0]], [data_feature[j][1]], mark[i], markersize = 5)
    j=j+1
plt.show()
