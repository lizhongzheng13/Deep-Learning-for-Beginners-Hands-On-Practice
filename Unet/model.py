# @Author : LiZhongzheng
# 开发时间  ：2025-02-28 21:09
import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            # padding_mode='reflect'取对称，加强特征提取的能力
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, channel):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, channel):
        super(Up, self).__init__()

        self.up_conv = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        # 使用插值法
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.up_conv(up)
        return torch.cat([out, feature_map], dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(3, 64)
        self.down1 = Down(64)
        self.conv2 = DoubleConv(64, 128)
        self.down2 = Down(128)
        self.conv3 = DoubleConv(128, 256)
        self.down3 = Down(256)
        self.conv4 = DoubleConv(256, 512)
        self.down4 = Down(512)
        self.conv5 = DoubleConv(512, 1024)
        self.up1 = Up(1024)
        self.conv6 = DoubleConv(1024, 512)
        self.up2 = Up(512)
        self.conv7 = DoubleConv(512, 256)
        self.up3 = Up(256)
        self.conv8 = DoubleConv(256, 128)
        self.up4 = Up(128)
        self.conv9 = DoubleConv(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        r1 = self.conv1(x)
        r2 = self.conv2(self.down1(r1))
        r3 = self.conv3(self.down2(r2))
        r4 = self.conv4(self.down3(r3))
        r5 = self.conv5(self.down4(r4))
        u1 = self.conv6(self.up1(r5, r4))
        u2 = self.conv7(self.up2(u1, r3))
        u3 = self.conv8(self.up3(u2, r2))
        u4 = self.conv9(self.up4(u3, r1))

        return self.Th(self.out(u4))


if __name__ == '__main__':
    x = torch.randn((2, 3, 256, 256))
    model = UNet()
    print(model(x).shape)
