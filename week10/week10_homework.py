#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
# 在自定义的文件中进行导入
# from resnet import resnet18
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

batchsize = 1
total_epoch = 30
learn_rate = 0.0001
validation_split = .2
shuffle_dataset = True
random_seed = 42

# 数据集地址
img_path = './week10_dataset_v003/work/image/'
cmd_path = './week10_dataset_v003/work/'
label2numdist = {'农夫山泉': 0, '冰露': 1, '娃哈哈': 2,
                 '可口可乐': 3, '天府可乐': 4, '其它': 5,
                 '康师傅': 6, '百事可乐': 7, '怡宝': 8,
                 '百岁山': 9, '苏打水': 10, '景甜': 11,
                 '恒大冰泉': 12, '今麦郎': 13}
num2labeldist = {'0': '农夫山泉', '1': '冰露', '2': '娃哈哈',
                 '3': '可口可乐', '4': '天府可乐', '5': '其它',
                 '6': '康师傅', '7': '百事可乐', '8': '怡宝',
                 '9': '百岁山', '10': '苏打水', '11': '景甜',
                 '12': '恒大冰泉', '13': '今麦郎'}


def anno2txt_train(file_path):  # 用于从annotation文件提取所需信息，划分数据集，并存为txt文件
    file = open(file_path, 'r', encoding='utf-8')
    imgs = []
    for line in file:
        line = line.split(':')
        label_name = line[2].split(',')[0].strip('"')
        label = label2numdist.get(label_name)
        img = line[-2].split(',')[0].replace('//open-class-0319/weeek10-dataset/image/', '').rstrip('"')
        imgs.append([img, label])
    # 划分数据集
    val_size = int(validation_split * len(imgs))
    train_size = len(imgs) - val_size
    train_data, val_data = random_split(imgs, [train_size, val_size])
    # 写入txt文件
    ftrain = open('./week10_dataset_v003/work/train.txt', 'w', encoding='utf-8')
    fval = open('./week10_dataset_v003/work/val.txt', 'w', encoding='utf-8')
    for list_i in train_data:
        ftrain.write(",".join(str(x) for x in list_i) + '\n')
    ftrain.close()
    for list_i in val_data:
        fval.write(",".join(str(x) for x in list_i) + '\n')
    fval.close()


def anno2txt_test(file_path):  # 用于从annotation文件提取所需信息，并存为txt文件
    file = open(file_path, 'r', encoding='utf-8')
    imgs = []
    for line in file:
        line = line.split(':')
        label_name = line[2].split(',')[0].strip('"')
        label = label2numdist.get(label_name)
        img = line[-2].split(',')[0].replace('//open-class-0319/weeek10-dataset/image/', '').rstrip('"')
        imgs.append([img, label])
    # 写入txt文件
    ftest = open('./week10_dataset_v003/work/test.txt', 'w',encoding='utf-8')
    for list_i in imgs:
        ftest.write(",".join(str(x) for x in list_i) + '\n')
    ftest.close()

# 将标注文件的内容提取并保存为txt文件
# anno_file_path = './week10_dataset_v003/work/label/week10-dataset-TMEuQxwldvN9mIgArsX/annotation/V003/V003.manifest'
# anno_test_path = './week10_dataset_v003/work/test_.txt'  # 就是老师给的test.txt文件,名字改为test_.txt
# # 生成train、val数据集的txt文件
# anno2txt_train(anno_file_path)
# # 生成test数据集的txt文件
# anno2txt_test(anno_test_path)


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


# 创建自己的类：MyDataset, 继承 Dataset类
class MyDataset(Dataset):
    def __init__(self, txt, data_path=None, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__() # 对继承父类的属性初始化
        # 在__init__()方法中得到图像的路径，然后将图像路径组成一个数组
        file_path = data_path + txt
        file = open(file_path, 'r', encoding='utf-8')
        imgs = []
        for line in file:
            line = line.split(',')
            # print(line[0].rstrip(','))  # img
            # print(line[1].rstrip('\n'))  # label
            imgs.append((line[0].rstrip(' '), line[1].rstrip('\n').lstrip(' ')))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data_path = data_path

    def __getitem__(self, index):
        # 按照索引读取每个元素的具体内容
        imgName, label = self.imgs[index]
        imgPath = img_path + imgName
        img = self.loader(imgPath)
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
            if label != 'None':
                label = torch.from_numpy(np.array(int(label)))
        return img, label

    def __len__(self):
        # 数据集的图片数量
        return len(self.imgs)


# 图像初始化操作
normalize = transforms.Normalize(mean=[0.557, 0.517, 0.496], std=[0.210, 0.216, 0.222])
# 图像增强
train_transformer = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomResizedCrop((448),scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    normalize,
])
# val和test的transforms
test_transformer = transforms.Compose([
    transforms.Resize(448),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

# 数据集加载方式设置
train_data = MyDataset(txt='train.txt', data_path=cmd_path, transform=train_transformer)
val_data = MyDataset(txt='val.txt', data_path=cmd_path, transform=test_transformer)
test_data = MyDataset(txt='test.txt', data_path=cmd_path, transform=test_transformer)

# 调用DataLoader和数据集
train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=False, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=batchsize, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False, num_workers=4)

# 判断使用CPU还是GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#加载预训练模型
# 调用模型
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
# print(model)
# 提取fc层中固定的参数
fc_features = model.fc.in_features
# 修改类别为14
model.fc = nn.Linear(fc_features, 14)
model = model.to(device)

model_name = 'resnet18'
torch.cuda.empty_cache()

# 定义损失函数，二分类交叉信息熵
criteria=nn.CrossEntropyLoss()
# 定义优化器

optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.99))
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1, last_epoch=-1)

# 训练
def model_train(optimizer, epoch, model, train_loader, modelname, criteria):
    model.train()
    train_loss, train_correct = 0, 0

    for batch_index, (img, label) in enumerate(train_loader):
        # 将数据放入device中
        img_data, target_label = img.to(device), label.to(device)
        # 前向传播
        output = model(img_data)
        # 计算损失函数
        loss = criteria(output, target_label.long())
        # 积累损失
        train_loss += loss
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 更新学习率
        # scheduler.step()

        # 得到预测结果
        pred = output.argmax(dim=1, keepdim=True)
        # 累加预测与标签相等的次数
        train_correct += pred.eq(target_label.long().view_as(pred)).sum().item()

    # 显示训练结果
    print('训练集：平均loss：{:.4f},准确率：{}/{}({:.0f}%)'.format(train_loss / len(train_loader.dataset),
                                                        train_correct, len(train_loader.dataset),
                                                        100 * train_correct / len(train_loader.dataset)))
    # 返回一次epoch的平均误差
    return train_loss / len(train_loader.dataset),100 * train_correct / len(train_loader.dataset)


# 验证
def model_val(model, val_loader, criteria):
    # 需要进行eval
    model.eval()
    val_loss, correct = 0, 0

    # 不需要计算模型梯度
    with torch.no_grad():
        predlist, scorelist, targetlist = [], [], []
        # 预测
        for batch_index, (img, label) in enumerate(val_loader):
            img_data, target_label = img.to(device), label.to(device)
            # 前向传播
            output = model(img_data)
            val_loss += criteria(output, target_label.long())
            # 计算score,使用softmax，行的和为1
            score = F.softmax(output, dim=1)
            # pred 是一个batch
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_label.long().view_as(pred)).sum().item()

            # # 由Gpu放到Cpu中
            # predlist = np.append(predlist, pred.cpu().numpy())
            # scorelist = np.append(scorelist, score[:, pred.cpu().numpy()].cpu().numpy())
            # targetlist = np.append(targetlist, target_label.long().cpu().numpy())

        # 显示验证结果
        print('验证集：平均loss：{:.4f},准确率：{}/{}({:.0f}%)'.format(val_loss / len(val_loader.dataset),
                                                            correct, len(val_loader.dataset),
                                                            100 * correct / len(val_loader.dataset)))
    return val_loss / len(val_loader.dataset), 100 * correct / len(val_loader.dataset)


# 测试
def model_test(model, test_loader):
    # 需要进行eval
    model.eval()
    file_path = cmd_path + 'test.txt'
    file = open(file_path, 'r', encoding='utf-8')
    images = []
    for line in file:
        line = line.split(',')
        images.append(line[0].rstrip(' '))

    pre_label = []
    for batch_index, (img, _) in enumerate(test_loader):
        img = img.to(device)
        output = model(img)
        _, idx = torch.max(output.data, 1)  # 输出最大值的位置
        print("idx: ", idx.item())
        print('预测标签名称： ', num2labeldist[str(idx.item())])
        # 存储预测结果
        pre_label.append([images[batch_index], idx.item()])

        # 显示最终预测图片及结果
        path = img_path + images[batch_index]
        img = default_loader(path)
        plt.imshow(img)
        plt.show()
        plt.pause(5)
    return pre_label


if __name__ == '__main__':

    # 定义4个数组
    train_loss_list = []
    train_acc_list = []
    val_loss_list= []
    val_acc_list = []

    # 训练模型并保存
    for epoch in range(total_epoch):
        # 进行一次epoch
        print('epoch:', epoch+1)
        train_loss, train_acc = model_train(optimizer, epoch, model, train_loader, model_name, criteria)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # 用验证集来验证
        val_loss, val_acc = model_val(model, val_loader, criteria)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # 打印真实结果
        # print('Target:',targetlist)
        # 输出预测label=1的概率
        # print('Score:', scorelist)
        # 输出预测结果
        # print('Predict:',predlist)

        # 模型保存
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 'bottle_resnet18.pt')

    # 学习曲线
    epoch = range(1, total_epoch+1)
    plt.figure()
    plt.plot(epoch, train_loss_list, 'bo', label='Training loss') # 'bo'为画蓝色圆点，不连线
    plt.plot(epoch, val_loss_list, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend() # 绘制图例，默认在右上角
    plt.show()

    plt.figure()
    plt.plot(epoch, train_acc_list, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    plt.plot(epoch, val_acc_list, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()  # 绘制图例，默认在右上角
    plt.show()

    # 将训练模型加载到测试集，得到测试集预测结果
    # PATH = './bottle_resnet18.pt'
    # if os.path.exists(PATH):
    #     model = models.resnet18()
    #     # 提取fc层中固定的参数
    #     fc_features = model.fc.in_features
    #     # 修改类别为14
    #     model.fc = nn.Linear(fc_features, 14)
    #     model.load_state_dict(torch.load(PATH))
    #
    #     model = model.to(device)
    pre_result = model_test(model, test_loader)
    print(pre_result)



