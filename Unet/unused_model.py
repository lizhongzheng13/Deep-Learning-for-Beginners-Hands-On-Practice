# @Author : LiZhongzheng
# 开发时间  ：2025-02-27 8:17
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.ups = nn.ModuleList()  # 容器，用于存储上采样和下采样的操作或层
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # stride=2可以根据公式可计算

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=3, stride=1, padding=1))
            self.ups.append(DoubleConv(in_channel=feature * 2, out_channel=feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []  # 把卷积完的结果存储到这个中，为了上采样部分的拼接
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转列表

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index // 2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            # size = skip_connection.shape[2:]: skip_connection.shape[2:]提取的是skip_connection张量的空间维度（即height和width），
            # 并将其作为resize的目标尺寸。
            concat = torch.cat((x, skip_connection), dim=1)  # 拼接
            x = self.ups[index + 1](concat)
        return self.final_conv(x)


def test():
    x = torch.randn((3, 3, 160, 160))
    model = UNet(in_channel=3, out_channel=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == '__main__':
    test()
