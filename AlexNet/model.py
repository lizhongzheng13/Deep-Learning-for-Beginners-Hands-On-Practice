# @Author : LiZhongzheng
# 开发时间  ：2025-02-14 11:15
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


# 我们使用的例子是单通道的
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()  # 激活函数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.s2(x)
        x = self.ReLU(self.conv3(x))
        x = self.s4(x)
        x = self.ReLU(self.conv5(x))
        x = self.ReLU(self.conv6(x))
        x = self.ReLU(self.conv7(x))
        x = self.s8(x)
        x = self.flatten(x)

        x = self.ReLU(self.f1(x))
        # 让中间的神经元，50%的是失活的，防止过拟合（参数太多，防止过拟合）
        x = F.dropout(x, 0.5)
        x = self.ReLU(self.f2(x))
        x = F.dropout(x, 0.5)
        x = self.ReLU(self.f3(x))
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))

