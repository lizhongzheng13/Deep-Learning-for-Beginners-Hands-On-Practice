# @Author : LiZhongzheng
# 开发时间  ：2025-02-26 10:36
import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 权重文件路径、数据路径和保存路径
    weight_path = 'params/unet.pth'
    data_path = r'D:\BaiduNetdiskDownload\UnetData\VOCdevkit\VOC2012'
    save_path = 'train_image'

    num_classes = 3

    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = Unet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('成功加载权重！')
    else:
        print('未能加载权重')
    # 定义优化器和损失函数
    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()
    # 训练过程
    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 前向传播
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            # 反向传播和优化
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 每1次输出训练损失
            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # 保存图像
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        # 每20个epoch保存一次模型
        if epoch % 20 == 0:
            torch.save(net.state_dict(), weight_path)
            print('模型保存成功！')

        epoch += 1


if __name__ == '__main__':
    train_model()
