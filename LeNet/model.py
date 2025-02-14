# @Author : LiZhongzheng
# 开发时间  ：2025-02-11 14:11
import torch
from torch import nn  # 里面包含一些层
# torchsummary 不仅可以查看网络的顺序结构，还有网络参数量，网络模型大小等信息
from torchsummary import summary


class LeNet(nn.Module):
    # 含有初始化，网络层和激活函数
    # 每个神经元后面都要加上激活函数，来保证神经元适应复杂的非线性问题
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()  # 激活函数
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 平展层

        # 线性全连接层
        self.f5 = nn.Linear(in_features=400, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 注意，里面的x每次都是更新后的新的x
        x = self.sig(self.conv1(x))
        x = self.s2(x)
        x = self.sig(self.conv3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


# 主函数
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
