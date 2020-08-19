#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# KMeans
# step1：随机设定初始聚类中心
# step2：将距离某个聚类中心距离近的样本点归类到该聚类中心，将样本全部归类完毕后得到多个簇
# step3：计算每个簇的均值作为新的聚类中心
# step4：重复第二步和第三步直至聚类中心不再发生变化 或达到迭代次数

# KMeans++
# step1：首先从所有样本中随机选定第一个聚类中心
# step2：记录所有样本到与其最近的聚类中心的距离
# step3：所有非聚类中心样本点被选取作为下一个聚类中心的概率与step2中的距离大小成正比，也就是说距离越远的样本点越有可能成为下一个聚类中心
# step4：重复step2和step3直至选出多个聚类中心
from hct66 import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from mnist import MNIST
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

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
    return feature.reshape(1,-1)

class KMcluster():
    def __init__(self,X,y,n_clusters,initialize = 'random',max_iters = 10):
        self.X = X
        self.y = y
        self.n_clusters = n_clusters
        self.initialize = initialize
        self.max_iters = max_iters

    # 随机初始化聚类中心点
    def init_random(self):
        n_samples,n_features = self.X.shape
        centroids = self.X[np.random.choice(n_samples,self.n_clusters)]
        return centroids

    # KMeans++ 初始化聚类中心点
    def init_kmeans_pp(self):
        n_samples,n_features = self.X.shape
        #step1:随机选取第一个中心点
        centroids = self.X[np.random.choice(n_samples,1)]
        #step2：计算每个样本到每一个聚类中心的欧式距离
        for k in range(0,self.n_clusters-1):
            distances = np.zeros((n_samples,k+1))
            for i in range(len(centroids)):
                distances[:,i] = np.sqrt(np.sum(np.square(self.X - centroids[i]),axis=1))
            #step3:计算每个样本与最近聚类中心（即已选择的聚类中心）的距离D(x)
            dist = np.min(distances,axis=1)  # axis = 1 对比列值，选取最小值作为样本与所属类别的距离
            #step4：再取一个随机值，用权重的方式来取计算下一个“种子点”。
            # 具体实现是，先取一个能落在Sum(D(x))中的随机值Random，
            # 然后用Random -= D(x)，直到其<=0，此时的点就是下一个“种子点”。
            # 即哪个D(x)最大，最有机会被选出作为种子点
            total = np.sum(dist)*np.random.rand()
            for j in range(n_samples):
                total -= dist[j]
                if total > 0:
                    continue
                centroids = np.r_[centroids,[self.X[j]]] #numpy.r_是将一系列的序列合并到一个数组中 相当于append
                break
        return centroids

    # 按距离分配聚类中心
    def assignment(self,centroids):
        n_samples = self.X.shape[0]
        distances = np.zeros((n_samples,self.n_clusters))
        # 计算出每个样本与每个聚类中心点的距离
        for i in range(self.n_clusters):
            distances[:,i] = np.sum(np.square(self.X - centroids[i]),axis=1)
        # 按每个样本离哪个中心点距离最短 ，返回该中心点的id
        return np.argmin(distances,axis=1)

    # 更新聚类中心
    def update_center(self,flag,centroids):
        new_centroids = np.zeros_like(centroids)
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(self.X[flag == i],axis=0)
        return new_centroids

    def train(self):
        if self.initialize == 'kmeans++':
            centroids = self.init_kmeans_pp()
        else:
            centroids = self.init_random()
        colmap = [i for i in range(self.n_clusters)]
        for i in range(self.max_iters):
            flag = self.assignment(centroids)
            plt.scatter(self.X[3, 0],self.X[3, 1],marker='+',s=100)  # 标签为0
            plt.scatter(self.X[6, 0], self.X[6, 1], marker='+', s=200)  #标签为0   从图中看它们并没有被聚到一类
            plt.scatter(self.X[:, 0], self.X[:, 1], c=flag, marker=".", alpha=0.5)
            plt.scatter(centroids[:, 0], centroids[:, 1], c=colmap, marker="o", linewidths=6)
            plt.show()

            new_centroids = self.update_center(flag,centroids)
            # 终止条件，如果重新计算的中心点与上一次的重复，则退出训练
            if (new_centroids == centroids).all():
                break
            else:
                centroids = new_centroids
            print("iters:", i, "center point:",centroids)

if __name__ =="__main__":
    """
    # hct66 dataset
    image_data,image_label = generate_data()
    image_label = np.array(image_label)
    col,row = image_data[0].shape
    image_values = np.zeros((len(image_label),col*row))
    for idx,image_value in enumerate(image_data):
        image_values[idx,:] = image_value.flatten()
    image_values_norm = StandardScaler().fit_transform(image_values)
    model_1 = KMcluster(image_values_norm,image_label,n_clusters=3,initialize='kmeans++',max_iters=10)
    model_1.train()
    """
    # mnist dataset
    mndata = MNIST('../week4/mnist/python-mnist/data/')
    image_data_all, image_label_all = mndata.load_training()
    image_data_1000 = image_data_all[0:1000]
    image_label_1000 = image_label_all[0:1000]
    data_feature = [np.array(get_feature_hog(i)) for i in image_data_1000]
    image_values_1000 = np.zeros((len(image_label_1000),(data_feature[0].shape[0])*(data_feature[0].shape[1])))
    for idx,image_value in enumerate(data_feature):
        image_values_1000[idx,:] = image_value.flatten()
    image_values_norm = StandardScaler().fit_transform(image_values_1000)
    # PCA 降维
    pca = PCA(n_components=2)
    image_values_pca = pca.fit(image_values_norm).transform(image_values_norm)
    model_2 = KMcluster(image_values_pca, image_label_1000, n_clusters=10, initialize="kmeans++", max_iters=50)
    model_2.train()

