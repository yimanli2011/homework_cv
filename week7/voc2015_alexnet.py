#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# 数据的读取
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from alex_data_class import MyDataset,AlexNet
import os
import time
import torch
import torch.nn as nn
# import torch.optim as optim
# 导入用于记录训练过程的日志类
# from logger import Logger

# 若检测到GPU环境则使用GPU，否则使用CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = os.path.dirname(os.path.dirname(__file__))


# 定义数据的处理方式
data_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize(256),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(227),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# transform = transforms.Compose([
#     transforms.Resize((227,227)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),  #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
#                             # 注意事项：归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) #[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
# ])

#txt这里的路径是当前项目下
#这里labels2.txt里面的图片路径是绝对路径，而且一定要transform，防止出现图片大小不一，
#因为DataLoader只识别大小尺寸统一的图片
# 加载数据集
data_dir_train = './voc2005_1/Annotations/'
data_dir_val = './voc2005_2/Annotations/'
train_data=MyDataset(data_dir=data_dir_train, transform=data_transforms['train'],train=True)
train_loader = DataLoader(train_data, batch_size=6,shuffle=True)
val_data=MyDataset(data_dir=data_dir_val, transform=data_transforms['val'],train=False)
val_loader = DataLoader(val_data, batch_size=6,shuffle=True)
dataloaders = {'train': train_loader, 'val': val_loader}
# 读取数据集大小
dataset_sizes = {'train': train_data.__len__(), 'val': val_data.__len__()}
# print(len(data_loader))
# for i, data in enumerate(data_loader):
#      inputs, labels = data
#      print(inputs.shape,labels)

# define loss
net = AlexNet().to(device)  # 实例化网络，有GPU则将网络放入GPU加速
loss_fuc = nn.CrossEntropyLoss()  # 多分类问题，选择交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.001,momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.001)  # 选择Adam，学习率取0.001
#分别为训练集和测试集定义两个数组
loss_list_train = []
accuracy_list_train = []
loss_list_val = []
accuracy_list_val = []
# Start train
EPOCH = 16  # 训练总轮数
since = time.time()
for epoch in range(EPOCH):
    sum_loss_batch = 0
    sum_loss_train = 0
    total_train = 0
    correct_train = 0
    # 数据读取
    for i, data in enumerate(train_loader):
        train_inputs, train_labels = data
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)  # 有GPU则将数据置入GPU加速
        # 梯度清零
        optimizer.zero_grad()
        # 传递损失 + 更新参数
        output_train = net(train_inputs)
        # 输出得分最高的类
        _, predicted_train = torch.max(output_train.data, 1)
        loss = loss_fuc(output_train, train_labels)
        loss.backward()
        optimizer.step()
        # 每训练100个batch打印一次平均loss
        sum_loss_batch += loss.item()
        sum_loss_train += loss.item()
        total_train += train_labels.size(0)  # 统计图片的总个数
        correct_train += (predicted_train == train_labels).sum()  # 统计正确分类的个数
        if i % 100 == 99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss_batch / 100))
            sum_loss_batch = 0.0
    epoch_loss_train = sum_loss_train / total_train
    epoch_acc_train = torch.true_divide(correct_train, total_train)
    loss_list_train.append(epoch_loss_train)
    accuracy_list_train.append(epoch_acc_train)
    print('第%d个epoch训练集的识别准确率为：%d%%' % (epoch + 1, 100 * torch.true_divide(correct_train, total_train)))
    # 保存模型
    torch.save(net.state_dict(),os.path.join(path,"alexnet"+str(epoch+1)+"_weight.pth"))

    # 测试集
    sum_loss_val = 0
    correct_val = 0
    total_val = 0
    for data in val_loader:
        val_inputs, val_labels = data
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        outputs_val = net(val_inputs)
        loss = loss_fuc(outputs_val, val_labels)
        sum_loss_val += loss.item()
        _, predicted_val = torch.max(outputs_val.data, 1)  # 输出得分最高的类
        total_val += val_labels.size(0)  # 统计图片的总个数
        correct_val += (predicted_val == val_labels).sum()  # 统计正确分类的个数

    print('第%d个epoch测试集的识别准确率为：%d%%' % (epoch + 1, 100 * torch.true_divide(correct_val, total_val)))

    epoch_loss_val = sum_loss_val / total_val
    epoch_acc_val = torch.true_divide(correct_val, total_val)
    loss_list_val.append(epoch_loss_val)
    accuracy_list_val.append(epoch_acc_val)
# plot history
x1 = range(0, EPOCH)
x2 = range(0, EPOCH)
y1_1 = accuracy_list_train
y1_2 = accuracy_list_val
y2_1 = loss_list_train
y2_2 = loss_list_val
plt.subplot(2, 1, 1)
plt.plot(x1, y1_1, 'o-',label='accuracy_train', color='black')
plt.plot(x1, y1_2, 'o-',label='accuracy_val', color='green')
plt.title('accuracy vs. epoches')
plt.subplot(2, 1, 2)
plt.plot(x2, y2_1, '.-',label='loss_train', color='black')
plt.plot(x2, y2_2, '.-',label='loss_val', color='green')
plt.xlabel('loss vs. epoches')
plt.legend(loc='upper left')
plt.show()
plt.savefig("accuracy_loss.jpg")






#
# # 初始化路径来记录模型训练过程中训练阶段与验证阶段的loss变化
# # 训练阶段的日志文件存储路径
# train_log_path = r"./log/train_log"
# train_logger = Logger(train_log_path)
# # 验证阶段日志文件存储路径
# val_log_path = r"./log/val_log"
# val_logger = Logger(val_log_path)
#
# # 训练与验证网络（所有层都参加训练）
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 10)
#
#         # 每训练一个epoch，验证一下网络模型
#         for phase in ['train', 'val']:
#             running_loss = 0.0
#             running_corrects = 0.0
#             if phase == 'train':
#                 # 学习率更新方式
#                 scheduler.step()
#                 #  调用模型训练
#                 model.train()
#                 # 依次获取所有图像，参与模型训练或测试
#                 for data in dataloaders[phase]:
#                     # 获取输入
#                     inputs, labels = data
#                     # 判断是否使用gpu
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     # 梯度清零
#                     optimizer.zero_grad()
#                     # 网络前向运行
#                     outputs = model(inputs)
#                     # 获取模型预测结果
#                     _, preds = torch.max(outputs.data, 1)
#                     # 计算Loss值
#                     loss = criterion(outputs, labels)
#                     # 反传梯度
#                     loss.backward()
#                     # 更新权重
#                     optimizer.step()
#                     # 计算一个epoch的loss值
#                     running_loss += loss.item() * inputs.size(0)
#                     # 计算一个epoch的准确率
#                     running_corrects += torch.sum(preds == labels.data)
#                 # 计算Loss和准确率的均值
#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = float(running_corrects) / dataset_sizes[phase]
#                 print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#                 torch.save(model.state_dict(), 'The_' + str(epoch) + '_epoch_model.pkl')
#
#                 # 1. 记录这个epoch的loss值和准确率
#                 info = {'loss': epoch_loss, 'accuracy': epoch_acc}
#                 for tag, value in info.items():
#                     train_logger.scalar_summary(tag, value, epoch)
#
#                 # 2. 记录这个epoch的模型的参数和梯度
#                 for tag, value in model.named_parameters():
#                     tag = tag.replace('.', '/')
#                     train_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
#                     train_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
#
#                 # 3. 记录最后一个epoch的图像
#                 info = {'images': inputs.cpu().numpy()}
#
#                 for tag, images in info.items():
#                     train_logger.image_summary(tag, images, epoch)
#
#             else:
#                 # 取消验证阶段的梯度
#                 with torch.no_grad():
#                     # 调用模型测试
#                     model.eval()
#                     # 依次获取所有图像，参与模型训练或测试
#                     for data in dataloaders[phase]:
#                         # 获取输入
#                         inputs, labels = data
#                         # 判断是否使用gpu
#                         inputs, labels = inputs.to(device), labels.to(device)
#                         # 网络前向运行
#                         outputs = model(inputs)
#                         _, preds = torch.max(outputs.data, 1)
#                         # 计算Loss值
#                         loss = criterion(outputs, labels)
#                         # 计算一个epoch的loss值
#                         running_loss += loss.item() * inputs.size(0)
#                         # 计算一个epoch的准确率
#                         running_corrects += torch.sum(preds == labels.data)
#
#                     # 计算Loss和准确率的均值
#                     epoch_loss = running_loss / dataset_sizes[phase]
#                     epoch_acc = float(running_corrects) / dataset_sizes[phase]
#                     print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#                     # 1. 记录这个epoch的loss值和准确率
#                     info = {'loss': epoch_loss, 'accuracy': epoch_acc}
#                     for tag, value in info.items():
#                         val_logger.scalar_summary(tag, value, epoch)
#
#                     # 2. 记录这个epoch的模型的参数和梯度
#                     for tag, value in model.named_parameters():
#                         tag = tag.replace('.', '/')
#                         val_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
#                         val_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
#
#                     # 3. 记录最后一个epoch的图像
#                     info = {'images': inputs.cpu().numpy()}
#                     for tag, images in info.items():
#                         val_logger.image_summary(tag, images, epoch)
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#
#
#
# if __name__ == '__main__':
#
#     # # 精调AlexNet
#     # # 导入Pytorch封装的AlexNet网络模型  model下载于：C:\Users\42259\.cache\torch\hub\checkpoints
#     # model = models.alexnet(pretrained=True)
#     # # 获取最后一个全连接层的输入通道数
#     # num_input = model.classifier[6].in_features
#     # # 获取全连接层的网络结构
#     # feature_model = list(model.classifier.children())
#     # # 去掉原来的最后一层
#     # feature_model.pop()
#     # # 添加上适用于自己数据集的全连接层
#     # feature_model.append(nn.Linear(num_input, 4))
#     # # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
#     # # 还可以为网络添加新的层
#     # # 重新生成网络的后半部分
#     # model.classifier = nn.Sequential(*feature_model)
#     # # 定义损失函数
#     # criterion = nn.CrossEntropyLoss()
#     # # 为不同层设定不同的学习率
#     # fc_params = list(map(id, model.classifier[6].parameters()))
#     # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
#     # params = [{"params": base_params, "lr": 0.0001},
#     #           {"params": model.classifier[6].parameters(), "lr": 0.001}, ]
#     # optimizer_ft = torch.optim.SGD(params, momentum=0.9)
#
#     # 自定义AlexNet网络
#     model = AlexNet().to(device)  # 实例化网络，有GPU则将网络放入GPU加速
#     # 定义损失函数
#     criterion = nn.CrossEntropyLoss()
#     # 为不同层设定不同的学习率
#     fc_params = list(map(id, model.fc3.parameters()))
#     base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
#     params = [{"params": base_params, "lr": 0.0001},
#               {"params": model.fc3.parameters(), "lr": 0.001}, ]
#     optimizer_ft = torch.optim.SGD(params, momentum=0.9)
#
#     from torch.optim import lr_scheduler
#     # 定义学习率的更新方式，每5个epoch修改一次学习率
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
#     train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

# 运行结束后，在cmd中输入并执行：tensorboard --logdir=E:\ML\kaikeba\CV\week7\log --port=6006
# 在浏览器中输入http://localhost:6006即可实现可视化
# 参考https://blog.csdn.net/MiaoB226/article/details/104213709








