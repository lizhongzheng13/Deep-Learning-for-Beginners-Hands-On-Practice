# @Author : LiZhongzheng
# 开发时间  ：2025-02-24 20:38
import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    # 因为残差块中有的含有1x1的卷积，有的却没有，所以要加上一个参数进行分辨（use_1conv）
    # num_channels卷积核的数量
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        # 这里的填充一般设为1(padding=1)(conv2)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            # 在残差块（ResidualBlock）中，1×1卷积层的输出通道数并不一定必须与最后一个输出的通道数相同
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, padding=0,
                                   stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.ReLu(y + x)

        return y


class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            # in_channels=3,应该是因为使用的是rgb图的缘故
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # 注意:bn层里面放前一个通道数的大小
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, strides=1),
            Residual(64, 64, use_1conv=False, strides=1),
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, strides=2),
            Residual(128, 128, use_1conv=False, strides=1),
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, strides=2),
            Residual(256, 256, use_1conv=False, strides=1),
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, strides=2),
            Residual(512, 512, use_1conv=False, strides=1),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))
