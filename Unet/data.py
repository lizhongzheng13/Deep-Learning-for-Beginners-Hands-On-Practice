# @Author : LiZhongzheng
# 开发时间  ：2025-02-28 20:56
from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), ])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(self.path, "SegmentationClass"))
        print(len(self.name))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        segment_name = self.name[idx]
        segment_path = os.path.join(self.path, "SegmentationClass", segment_name)

        # image_path = os.path.join(self.path, "JPEGImages", segment_name.replace(".png", ".jpg"))
        image_path = os.path.join(self.path, "JPEGImages", segment_name)
        segment_image = keep_size(segment_path)
        image = keep_size(image_path)
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset('D:\\BaiduNetdiskDownload\\UnetData\\VOCdevkit\\VOC2012')
    print(data[0][0].shape)
    print(data[1][0].shape)
