#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, random_split
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import cv2
import random
import os
import math
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', **{'size': 12})
validation_split = .2

label2numdist = {'农夫山泉': 0, '冰露': 1, '娃哈哈': 2,
                 '可口可乐': 3, '天府可乐': 4, '其他': 5,
                 '康师傅': 6, '百事可乐': 7, '怡宝': 8,
                 '百岁山': 9, '苏打水': 10, '景甜': 11,
                 '恒大冰泉': 12, '今麦郎': 13}
num2labeldist = {'0': '农夫山泉', '1': '冰露', '2': '娃哈哈',
                 '3': '可口可乐', '4': '天府可乐', '5': '其他',
                 '6': '康师傅', '7': '百事可乐', '8': '怡宝',
                 '9': '百岁山', '10': '苏打水', '11': '景甜',
                 '12': '恒大冰泉', '13': '今麦郎'}
classes = ['农夫山泉', '冰露', '娃哈哈', '可口可乐', '天府可乐', '其他', '康师傅', '百事可乐', '怡宝',
           '百岁山', '苏打水', '景甜', '恒大冰泉', '今麦郎']


def anno2txt_train(file_path):  # 用于从annotation文件提取所需信息，划分数据集，并存为txt文件
    file = open(file_path, 'r', encoding='utf-8')
    imgs = []
    for line in file:
        line = line.split(':')
        # label_name = line[2].split(',')[0].strip('"')
        # label = label2numdist.get(label_name)
        img = line[-2].split(',')[0].replace('//open-class-0319/weeek10-dataset/image/', '').rstrip('"')
        imgs.append(img)
    # 划分数据集
    val_size = int(validation_split * len(imgs))
    train_size = len(imgs) - val_size
    train_data, val_data = random_split(imgs, [train_size, val_size])
    # 写入txt文件
    ftrain = open('./V002_dataset/train.txt', 'w', encoding='utf-8')
    fval = open('./V002_dataset/val.txt', 'w', encoding='utf-8')
    for list_i in train_data:
        ftrain.write(list_i + '\n')
    ftrain.close()
    for list_i in val_data:
        fval.write(list_i + '\n')
    fval.close()


def anno2txt_test(file_path):  # 用于从annotation文件提取所需信息，并存为txt文件
    file = open(file_path, 'r', encoding='utf-8')
    imgs = []
    for line in file:
        line = line.split(':')
        # label_name = line[2].split(',')[0].strip('"')
        # label = label2numdist.get(label_name)
        img = line[-2].split(',')[0].replace('//open-class-0319/weeek10-dataset/image/', '').rstrip('"')
        imgs.append(img)
    # 写入txt文件
    ftest = open('./V002_dataset/test.txt', 'w', encoding='utf-8')
    for list_i in imgs:
        ftest.write(list_i + '\n')
    ftest.close()


# # 将标注文件的内容提取并保存为txt文件
# anno_file_path = './V002_dataset/V002.manifest'
# anno_test_path = './V002_dataset/test.manifest'
# # 生成train、val数据集的txt文件
# anno2txt_train(anno_file_path)
# # # 生成test数据集的txt文件
# anno2txt_test(anno_test_path)


# 进行归一化操作
def convert(size, box): # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1./size[0]     # 1/w
    dh = 1./size[1]     # 1/h
    x = (box[0] + box[1])/2.0   # 物体在图中的中心点x坐标
    y = (box[2] + box[3])/2.0   # 物体在图中的中心点y坐标
    w = box[1] - box[0]         # 物体实际像素宽度
    h = box[3] - box[2]         # 物体实际像素高度
    x = x*dw    # 物体中心点x的坐标比(相当于 x/原图w)
    w = w*dw    # 物体宽度的宽度比(相当于 w/原图w)
    y = y*dh    # 物体中心点y的坐标比(相当于 y/原图h)
    h = h*dh    # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)    # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


def convert_annotation(image_id):
    '''
    将对应文件名的xml文件转化为label文件，xml文件包含了对应的bounding框以及图片长款大小等信息，
    通过对其解析，然后进行归一化最终读到label文件中去，也就是说
    一张图片文件对应一个xml文件，然后通过解析和归一化，能够将对应的信息保存到唯一一个label文件中去
    labal文件中的格式：class x y w h　　同时，一张图片对应的类别有多个，所以对应的ｂｕｎｄｉｎｇ的信息也有多个
    '''
    # 对应的路径找到相应的文件夹，并且打开相应image_id的xml文件，其对应bound文件
    in_file = open('V002_dataset/annotations/%s.xml' % (image_id), encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open('V002_dataset/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对
    root = tree.getroot()
    # 获得图片的尺寸大小
    size = root.find('size')
    # 获得宽
    w = int(size.find('width').text)
    # 获得高
    h = int(size.find('height').text)
    # 遍历目标obj
    for obj in root.iter('object'):
        # 获得difficult ？？
        difficult = obj.find('difficult').text
        # 获得类别 =string 类型
        cls = obj.find('name').text
        # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
        if cls not in classes or int(difficult) == 1:
            continue
        # 通过类别名称找到id
        cls_id = classes.index(cls)
        # 找到bndbox 对象
        xmlbox = obj.find('bndbox')
        # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        print(image_id, cls, b)
        # 带入进行归一化操作
        # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
        bb = convert((w, h), b)
        # bb 对应的是归一化后的(x,y,w,h)
        # 生成 class x y w h 在label文件中
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# train_file = './V002_dataset/train.txt'
# file = open(train_file, 'r', encoding='utf-8')
# for line in file:
#     image_path = line.split('.')[0]
#     convert_annotation(image_path)


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, rect=True, image_weights=False):
        with open(path, 'r', encoding='utf-8') as f:
            img_files = f.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, img_files))

        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.label_files = [x.replace('images', 'labels').
                                replace('.jpeg', '.txt').
                                replace('.jpg', '.txt').
                                replace('.bmp', '.txt').
                                replace('.png', '.txt') for x in self.img_files]

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5))] * n
        iter = tqdm(self.label_files, desc='Reading labels') if n > 1000 else self.label_files
        for i, file in enumerate(iter):
            try:
                file = 'V002_dataset/labels/' + file
                with open(file, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    if l.shape[0]:
                        assert l.shape[1] == 5, '> 5 label columns: %s' % file
                        assert (l >= 0).all(), 'negative labels: %s' % file
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                        self.labels[i] = l
            except:
                pass  # print('Warning: missing labels for %s' % self.img_files[i])  # missing label file
        assert len(np.concatenate(self.labels, 0)) > 0, 'No labels found. Incorrect label paths provided.'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = 'V002_dataset/image/' + self.img_files[index]
        label_path = 'V002_dataset/labels/' + self.label_files[index]

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'File Not Found ' + img_path
            if self.n < 1001:
                self.imgs[index] = img  # cache image into memory

        # Augment colorspace
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = (random.random() * 2 - 1) * fraction + 1
            b = (random.random() * 2 - 1) * fraction + 1
            S *= a
            V *= b

            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # Letterbox
        h, w, _ = img.shape
        if self.rect:
            shape = self.batch_shapes[self.batch[index]]
            img, ratio, padw, padh = letterbox(img, new_shape=shape, mode='rect')
        else:
            shape = self.img_size
            img, ratio, padw, padh = letterbox(img, new_shape=shape, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            # with open(label_path, 'r') as f:
            #     x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.Tensor(weights)


def plot_images(imgs, targets, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close()


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets, model):  # predictions, targets, model

    pass

# 定义网络
class MYNet(nn.Module):
    def __init__(self):
        super(MYNet, self).__init__()
        self.features = nn.Sequential(resnet18().conv1,
                                      resnet18().bn1,
                                      resnet18().relu,
                                      resnet18().maxpool,
                                      resnet18().layer1,
                                      resnet18().layer2,
                                      resnet18().layer3,)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        # 决策层：检测层
        self.detector = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1176),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        b, _ = x.shape
        x = x.view(b, 7, 7, 24)
        return x


train_path = 'V002_dataset/train.txt'
# 超参数设置
learn_rate = 0.0001
num_epoches = 70
img_size = 448
batch_size = 1
iou_thred = 0.5
# Dataset
dataset = LoadImagesAndLabels(train_path, img_size, batch_size, augment=True, rect=False)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                        shuffle=True, pin_memory=True, collate_fn=dataset.collate_fn)
# 判断使用CPU还是GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 定义模型
model = MYNet()
model = model.to(device)
# 定义优化器
# optimizer = optim.SGD(my_resnet.parameters(), lr=learn_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.99))
# # 学习率的调整
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
#                                                  factor=0.1, patience=3,
#                                                  verbose=True, threshold=1e-4)
model.class_weights = labels_to_class_weights(dataset.labels, 14).to(device)  # attach class weights


def train():
    latest = 'latest.pt'
    best = 'best.pt'
    nb = len(dataloader)
    maps = np.zeros(14)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t, t0 = time.time(), time.time()
    for epoch in range(num_epoches):
        print(f"epoch: {epoch + 1}")
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'targets', 'time'))

        # # Update scheduler
        # scheduler.step()

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname='train_batch%g.jpg' % i)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, num_epoches - 1),
                '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
            t = time.time()
            print(s)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        if epoch % 10 == 0:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch + 1, dt))
    return results


if __name__ == '__main__':
    # mynet = MYNet()
    # x = torch.randn(1, 3, 448, 448)
    # feature = mynet(x)
    # # feature_size b*7*7*30
    # print(feature.shape)
    train()