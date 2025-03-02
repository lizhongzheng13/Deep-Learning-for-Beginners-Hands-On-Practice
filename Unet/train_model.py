# @Author : LiZhongzheng
# 开发时间  ：2025-02-28 21:46

from torch import nn, optim
import torch
from data import *
from model import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = "params/weight.pth"
    data_path = r"D:\BaiduNetdiskDownload\UnetData\VOCdevkit\VOC2012"
    save_img_path = r'./train_img'

    # Create the directory if it doesn't exist
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("successfully load weight")
    else:
        print("no weight")

    opt = optim.Adam(net.parameters())
    loss_func = nn.BCELoss()

    epoch = 1
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            output = net(image)
            train_loss = loss_func(output, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i % 2 == 0:
                torch.save(net.state_dict(), weight_path)

            _img = image[0]
            _seg = segment_image[0]
            _out_img = output[0]

            img = torch.stack([_img, _seg, _out_img], dim=0)  # 三张图拼接
            save_image(img, f'{save_img_path}/{i}.png')

        epoch += 1


if __name__ == '__main__':
    train()
