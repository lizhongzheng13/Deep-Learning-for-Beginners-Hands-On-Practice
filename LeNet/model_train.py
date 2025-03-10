# @Author : LiZhongzheng
# 开发时间  ：2025-02-11 16:14
import copy
# from datetime import time
import time
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd


# 处理训练集和验证集
def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)  # size=224->28
    # 划分训练集和数据集
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)
    return train_dataloader, val_dataloader


train_dataloader, val_dataloader = train_val_data_process()


# 模型训练函数
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 优化器：梯度下降的优化版
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失函数：一般用于分类任务中~
    criterion = nn.CrossEntropyLoss()
    # 将模型放到训练设备中
    model = model.to(device)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高标准度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证机损失列表
    val_loss_all = []
    # 训练集准度列表
    train_acc_all = []
    # 验证机准度列表
    val_acc_all = []
    # 保存时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数 损失函数，精确度，训练的轮次
        train_loss = 0.0
        train_correct = 0.0

        val_loss = 0.0
        val_correct = 0.0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征和标签放到我们的设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 开启模型的训练模式
            model.train()

            # 向前传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为零：防止梯度为零
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对每一轮的损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_correct += torch.sum(pre_lab == b_y.data)
            # 训练集的样本数
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征和标签放到我们的设备中(验证集中~)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 开启验证模式
            model.eval()
            # 向前传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_correct += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        # 计算每次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.double().item() / val_num)

        print('{} Train loss:{:.4f} Train Acc;{:.4f} '.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val loss:{:.4f} Val Acc;{:.4f} '.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            # 保存当前最高的精准度
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print('训练耗费时间：{:.0f}m{:.0f}s'.format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数
    # pth是权重模型的后缀
    # model.state_dict(best_model_wts)
    torch.save(best_model_wts,
               'D:/python-mangshe/pythonProject1/深度学习入门教程/LeNet/best_model.pth')

    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "val_acc_all": val_acc_all
    })

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label='train loss')
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.show()


if __name__ == '__main__':
    # 将我们的模型实例化
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
