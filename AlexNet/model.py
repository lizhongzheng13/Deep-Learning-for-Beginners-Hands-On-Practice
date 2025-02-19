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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
