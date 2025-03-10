# @Author : LiZhongzheng
# 开发时间  ：2025-03-04 11:36
import swanlab
import random

# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-awesome-project",

    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    }
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # 记录训练指标
    swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()