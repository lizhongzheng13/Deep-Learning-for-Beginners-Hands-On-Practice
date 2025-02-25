# @Author : LiZhongzheng
# 开发时间  ：2025-02-11 14:54
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 加载 FashionMNIST 数据集
# transforms.ToTensor()将图像从 PIL 图像格式或 NumPy 数组格式转换为 PyTorch 的 Tensor 格式，并且将像素值归一化到 [0, 1] 范围内（即将像素值除以 255）
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

# shuffle=True打乱数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

# 获得一个Batch数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

# b_x.squeeze()：squeeze() 方法用于移除张量中大小为 1 的维度。
# 例如，如果 b_x 的形状是 [1, 224, 224]，经过 squeeze() 后会变成 [224, 224]
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换为Numpy数组(矩阵的格式)
batch_y = b_y.numpy()  # 将张量转为Numpy数组
class_label = train_data.classes  # 训练集的标签
# print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
