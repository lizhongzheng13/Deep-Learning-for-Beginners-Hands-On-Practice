# 深度学习实战

欢迎来到我的深度学习实战项目！本项目旨在通过实际案例和代码实现，帮助初学者快速入门深度学习。无论你是刚刚开始接触深度学习，还是希望在实践中巩固知识，这里都有适合你的内容。

## 项目简介

在这个项目中，我将通过一系列实战案例，涵盖从基础到进阶的深度学习任务，包括但不限于：

- 图像分类
- 数据预处理
- 模型训练与优化
- 模型评估与测试

**注意：**

- 对于每个模型的中的best_model.pth因文件过大而未上传，可通过运行model_train.py可以生成。
- AlexNet,GoogLeNet,LeNet,VGG16的data数据也没有上传，但是代码中有体现，会自动下载，大可不必担心。
- 猫狗分类模型中，使用的是自己的数据集，因为数据集太大不太好上传，大家可以从网上找找，比较容易找到，将数据集的名字可以改为data_cat_dog，里面的子文件分别是cat，dog。

**超级注意**

- 在有些模型中可能使用训练集和测试集的精准度很高,但是验证集的精准度比较低,可能的原因是
  `train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])`
  未加上最后的那个归一化处理,我的模型中就存在此问题,我的模型有些没有添加,可以自行加上去(位置是在model_train.py文件中)

我将使用 PyTorch 框架进行实现，但也会尽量保持代码的通用性和可读性，以便你可以轻松迁移到其他框架。

个人声明
我是一个深度学习初学者，希望通过这个项目记录自己的学习过程，并与更多志同道合的朋友交流。如果你有任何建议或意见，欢迎随时提出，我会非常感激！

联系我

- GitHub: https://github.com/lizhongzheng13?tab=repositories
- Email: 878954714@qq.com

<hr/>
特别感谢
感谢所有开源社区的贡献者，你们的代码和教程让我受益匪浅。特别感谢 PyTorch 社区提供的优秀框架和文档。最后特别感谢炮哥，让我对深度学习有所了解。
