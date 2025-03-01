# @Author : LiZhongzheng
# 开发时间  ：2025-02-28 21:03
# 进行等比缩放，以减少图像损失
from PIL import Image


def keep_size(path, size=(256, 256)):
    img = Image.open(path)
    max_len = max(size)
    mask = Image.new('RGB', (max_len, max_len), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask.resize(size)
    return mask
