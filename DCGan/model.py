# @Author : LiZhongzheng
# 开发时间  ：2025-03-03 18:26
import torch
import torch.nn as nn
from torchsummary import summary


# 生成器
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz  # 噪声向量Z
        self.ngf = ngf  # 通道数
        self.nc = nc  # 输出的通道数
        self.main = nn.Sequential(
            # 输入噪声向量Z，(ngf*8) x 4 x 4特征图
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入(ngf*8) x 4 x 4特征图，输出(ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入(ngf*4) x 8 x 8，输出(ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入(ngf*2) x 16 x 16，输出(ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输入(ngf) x 32 x 32，输出(nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.ndf = ndf  # 判别器特征图通道数量单位
        self.nc = nc  # 图片的通道数
        self.main = nn.Sequential(
            # 输入图片大小 (nc) x 64 x 64，输出 (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入(ndf) x 32 x 32，输出(ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入(ndf*2) x 16 x 16，输出 (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入(ndf*4) x 8 x 8，输出(ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输入(ndf*8) x 4 x 4，输出1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCGan(nn.Module):
    def __init__(self, generator, discriminator):
        super(DCGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, noise):
        gen_imgs = self.generator(noise)
        disc_imgs = self.discriminator(gen_imgs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建生成器和判别器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 测试生成器
    print("Generator Summary:")
    summary(generator, (100, 1, 1))  # 生成器的输入是噪声向量，形状为 (100, 1, 1)

    # 测试判别器
    print("\nDiscriminator Summary:")
    summary(discriminator, (3, 64, 64))  # 判别器的输入是图像，形状为 (3, 64, 64)