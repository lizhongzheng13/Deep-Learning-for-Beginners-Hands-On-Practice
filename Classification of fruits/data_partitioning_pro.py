# @Author : LiZhongzheng
# 开发时间  ：2025-02-21 21:07
import os
from shutil import copy
import random


# 用于创建文件夹
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 数据集划分函数
def split_dataset(file_path, train_dir='data/train', test_dir='data/test', split_rate=0.1):
    # 获取所有类名（即文件夹名）
    flower_class = [cla for cla in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, cla))]

    # 创建训练集和验证集文件夹
    mkfile(train_dir)
    mkfile(test_dir)

    for cla in flower_class:
        mkfile(os.path.join(train_dir, cla))
        mkfile(os.path.join(test_dir, cla))

    # 遍历所有类别，将图像分配到训练集和验证集
    for cla in flower_class:
        cla_path = os.path.join(file_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        # 如果当前类别下没有图片，则跳过
        if num == 0:
            print(f"警告：类别 {cla} 下没有图像文件，跳过该类别。")
            continue

        # 按照给定比例随机抽取验证集的图像
        eval_index = random.sample(images, k=int(num * split_rate))

        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)

            # 根据是否在验证集中的判断，将图片复制到训练集或验证集
            if image in eval_index:
                new_path = os.path.join(test_dir, cla)
            else:
                new_path = os.path.join(train_dir, cla)

            try:
                copy(image_path, new_path)
            except Exception as e:
                print(f"错误: 无法复制 {image_path} 到 {new_path}. 错误详情: {e}")

            print(f"\r[{cla}] 处理进度: [{index + 1}/{num}] 完成", end="")
        print()  # 打印换行，完成当前类别的处理

    print("数据集划分完成!")


if __name__ == '__main__':
    # 设置原始数据集路径和划分后的目标路径
    file_path = 'fruits'
    split_dataset(file_path)
