# @Author : LiZhongzheng
# 开发时间  ：2025-02-18 22:36
import torch
from torch import nn
from torchsummary import summary


# inception的作用：增加网络深度和宽度的同时减少参数
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 路线1,单1*1卷积层
        # c1卷积核的数量
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路线2,1*1卷积,3*3卷积
        # c2有两个卷积,则有两个输出
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=in_channels, out_channels=c2[1], kernel_size=3)

        # 路线3,1*1卷积,5*5卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=in_channels, out_channels=c3[1], kernel_size=5)
        
