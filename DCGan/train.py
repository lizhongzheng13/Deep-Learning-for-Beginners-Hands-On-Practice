# @Author : LiZhongzheng
# 开发时间  ：2025-03-03 18:52
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import *
import os, sys
import shutil
from tensorboardX import SummaryWriter

# 创建日志目录
writer = SummaryWriter('logs')  # 默认目录为 'logs'，用于 TensorBoard 记录训练过程
if os.path.exists("out"):
    print("删除 out 文件夹！")
    if sys.platform.startswith("win"):
        shutil.rmtree("./out")
    else:
        os.system("rm -r ./out")

print("创建 out 文件夹！")
os.mkdir("./out")
# 存储生成图片
imgs_path = "./imgs_output"

# 固定随机种子：确保实验的可重复性。
regular_seed = 999  # 999无任何含义，只是表明采取的是固定随机种子
random.seed(regular_seed)
torch.manual_seed(regular_seed)

# 参数设置
data_path = "D:\\BaiduNetdiskDownload\\DCGAN人脸嘴部表情生成\\data"
num_work = 0  # 进程数量选择 #0:数据加载完全在主进程中串行执行
batch_size = 16  # 每批次的图像数量
img_size = 64
img_channel = 3  # 图片的通道数（RGB = 3）
noise_dim = 100  # 生成器输入噪声维度
gen_out_channel = 64  # 生成器的输出通道
dis_in_channel = 64  # 判别器的输入通道

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gen = Generator().to(device)
model_dis = Discriminator().to(device)

# 定义损失函数与优化器
criterion = nn.BCELoss()  # 使用二元交叉熵损失
real_label = 1.0  # 告诉判别器，这个图片是真实数据集中的图片。
fake_label = 0.0  # 告诉判别器，这个图片是生成的，不是真实数据。

lr = 0.0003  # learning rate
beta1 = 0.5  # 常规 Adam 默认 beta1=0.9，但在 GAN 训练中，beta1=0.5 更常见，能缓解训练不稳定的问题。
opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, 0.999))
opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1, 0.999))

# 缓存生成结果
img_list = []
# 损失变量
G_losses = []
D_losses = []
# batch变量
iters = 0

# 读取数据
dataset = dset.ImageFolder(root=data_path, transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))  # 居中裁剪 CenterCrop(image_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_work)

# 训练循环
num_epochs = 50
fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)  # fixed_noise 是一个固定噪声向量，用于生成一致的样本
for epoch in range(num_epochs):
    loss_gen = 0.0
    loss_dis = 0.0
    for i, data in enumerate(dataloader, 0):  # 从索引 0 开始为每个批次的数据添加索引
        # 训练判别器 (D)
        # 判别真实图片
        model_dis.zero_grad()
        real_data = data[0].to(device)
        b_size = real_data.size(0)  # 当前批次中样本的数量
        label = torch.full((b_size,), real_label, device=device)  # 创建一个与当前批次大小（b_size）匹配的张量，其所有元素都填充为 real_label 的值
        output = model_dis(real_data).view(-1)  # .view(-1) 将其展平为一个形状为 (batch_size,) 的一维张量
        dis_real_loss = criterion(output, label)
        dis_real_loss.backward()
        D_real = output.mean().item()  # D_real 表示判别器对真实图像的平均输出值

        # 判别生成图片
        noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
        fake_img = model_gen(noise)
        label.fill_(fake_label)  # 将 label 这个张量的所有元素填充为 fake_label，即 0.0
        output = model_dis(fake_img.detach()).view(-1)
        dis_fake_loss = criterion(output, label)
        dis_fake_loss.backward()  # 计算梯度
        D_fake = output.mean().item()

        errD = dis_real_loss + dis_fake_loss
        opt_dis.step()  # 参数更新

        # 训练生成器(G)：让 G 生成的假图骗过 D
        model_gen.zero_grad()
        label.fill_(real_label)
        output = model_dis(fake_img).view(-1)
        errG = criterion(output, label)
        errG.backward()
        G_fake = output.mean().item()
        opt_gen.step()

        # 输出训练状态
        if i % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_real, D_fake, G_fake))
        # 存储损失
        loss_gen = loss_gen + errG.item()  ## 累加batch损失
        loss_dis = loss_dis + errD.item()  ## 累加batch损失

        # 结果可视化
        # 对固定的噪声向量，存储生成的结果
        if (iters % 20 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = model_gen(fixed_noise).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            i = vutils.make_grid(fake, padding=2, normalize=True)
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(i, (1, 2, 0)))
            plt.axis('off')  # 关闭坐标轴
            plt.savefig("out/%d_%d.png" % (epoch, iters))
            plt.close(fig)
        iters += 1  ## nbatch+1
    writer.add_scalar('data/lossG', loss_gen, epoch)
    writer.add_scalar('data/lossD', loss_dis, epoch)

torch.save(model_gen, 'models/model_gen.pth')
torch.save(model_dis, 'models/model_dis.pth')
