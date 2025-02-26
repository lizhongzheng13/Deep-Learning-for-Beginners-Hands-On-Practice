# @Author : LiZhongzheng
# 开发时间  ：2025-02-25 23:37
# 因为图片大小不同，所以图片需要进行缩放,等比缩放
from PIL import Image


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    # 获取当前图像的最长边
    temp = max(img.size)
    # 创建了一张全黑的图片，大小是temp × temp像素。这张图片没有任何图案，完全是黑色的。
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))  # 讲原图粘到这个黑色的幕布上
    mask = mask.resize(size)  # 使所有图像大小一致
    return mask
