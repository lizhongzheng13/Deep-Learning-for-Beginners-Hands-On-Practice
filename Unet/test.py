# @Author : LiZhongzheng
# 开发时间  ：2025-02-26 11:17
import torch
from torchvision.utils import save_image

from net import *
import os

from 深度学习入门.Unet.data import transfrom
from 深度学习入门.Unet.utils import keep_image_size_open

net = Unet().cuda()

weights = "params/unet.pth"
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully load weights')
else:
    print('weights not exist')

_input = input('please input image path:')
img = keep_image_size_open(_input)
img_data = transfrom(img).cuda()
print(img_data.shape)
img_data = torch.unsqueeze(img_data, 0)
out = net(img_data)
save_image(out, 'result/result.jpg')
# print(out)
