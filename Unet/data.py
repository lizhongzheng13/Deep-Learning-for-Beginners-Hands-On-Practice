# @Author : LiZhongzheng
# 开发时间  ：2025-02-25 23:20
import os

from opt_einsum.backends import torch
from torch.utils.data import Dataset
from utils import keep_image_size_open
from torchvision import transforms

transfrom = transforms.Compose([
    transforms.ToTensor(),
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 获取文件夹中的所有文件名
        self.name = os.listdir(os.path.join(path, "SegmentationClass"))

    # 获取数据集长度
    def __len__(self):
        return len(self.name)  # 返回数据集中样本的数量

    # 数据的制作
    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png当前的格式
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        # 原图的地址
        image_path = os.path.join(self.path, "JPEGImages", segment_name.replace('png', 'jpg'))  # 因为图片的格式不同，再需要转换一下
        # 因为图片大小不同，所以图片需要进行缩放————>utils.py
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transfrom(image), transfrom(segment_image)


if __name__ == '__main__':
    data = MyDataset("D:\\BaiduNetdiskDownload\\UnetData\\VOCdevkit\\VOC2012")
    print(data[0][0].shape)
    print(data[0][1].shape)
