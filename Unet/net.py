# @Author : LiZhongzheng
# 开发时间  ：2025-02-26 8:02
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # padding_mode='reflect'：使用反射填充（reflect padding）而不是默认的零填充（zero padding）。
            # 反射填充可以更好地保留图像边缘信息，减少边界效应，这在处理图像数据时可能会带来更好的效果。
            # bias = False：在卷积层中禁用了偏置项（bias）。
            # 这在某些情况下是为了配合批量归一化（BatchNorm）使用，
            # 因为BatchNorm已经包含了偏置调整的功能，再使用卷积层的偏置可能会导致冗余。
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样:用于减小特征图的空间维度（即高度和宽度），同时保留重要的特征信息。
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # 原本要使用最大池化做，但是最大池化丢特征才多了，则选择卷积
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channels),
            # 改进的ReLU激活函数，允许负值通过，避免神经元完全死亡。
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 上采样
class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channels, channels // 2, 1, 1)

    def forward(self, x, feature_map):
        # 使用F.interpolate对输入特征图x进行上采样（放大）
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        # 将两个特征图out和feature_map沿通道维度（dim = 1）拼接在一起
        return torch.cat((out, feature_map), dim=1)


class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        # self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        return self.out(O4)
        # 特征图 O4 的通道数为 64，而最终的预测结果需要有 num_classes 个通道，每个通道对应一个类别。
        # self.out 的作用是将 64 通道的特征图映射到 num_classes 通道的预测结果。


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 256, 256).to(device)  # 两个批次，3个通道，图像的大小为 256×256 像素
    net = Unet(num_classes=3).to(device)
    print(net(x).shape)
