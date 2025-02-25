# @Author : LiZhongzheng
# 开发时间  ：2025-02-14 9:22
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import ResNet18, Residual
from PIL import Image


def test_data_process():
    # 使用验证集
    # 定义数据集的路径
    ROOT_TRAIN = r'data\test'

    normalize = transforms.Normalize(mean=[0.0420662, 0.04281093, 0.04413987], std=[0.03315472, 0.03433457, 0.03628447])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader


# test_loader = test_data_process()


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 初始化参数
    # test_correct = 0.0只有改为张量形式下面才可以使用.double()方法
    test_num = 0
    test_corrects = 0.0
    # test_correct = torch.tensor(0.0, dtype=torch.float32)  # 初始化为张量

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            # 找到张量（Tensor）中最大值的索引
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本累加
            test_num += test_data_x.size(0)
    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率：", test_acc)


if __name__ == '__main__':
    model = ResNet18(Residual)
    # 加载模型权重
    model.load_state_dict(torch.load("best_model.pth"))

    ###############################################################################################
    test_loader = test_data_process()
    test_model_process(model, test_loader)
    ###############################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    classes = ["艾保利奥米", "巴斯马蒂香米", "伊普萨拉米", "茉莉香米", "卡拉贾山米"]

    ###############################################################################################
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)

            pre_label = torch.argmax(output, dim=1)
            # 将预测标签和真实标签从张量转换为Python的标量值（整数）
            result = pre_label.item()
            label = b_y.item()
            print("预测值:", classes[result], '---------', '真实值:', classes[label])

    ###############################################################################################

    # # 这里相当于从网上找个图片实际验证下预测到底如何（相当于模型的实际测试）
    # image = Image.open('test01.png')
    # # print(image.size)
    # # 图形预处理：尺寸变化；格式转换（变为Tensor格式）；归一化
    # normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155, 0.06216329, 0.05930814])
    # test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # image = test_transform(image)
    # # print(image.shape)
    #
    # # 注意点
    # # 当前数据的维度是3*224*224，但是还需要添加一个批次的维度才能送入到模型中
    # # 添加批次维度
    # image = image.unsqueeze(0)
    # # print(image.shape)
    #
    # with torch.no_grad():
    #     model.eval()
    #     image = image.to(device)
    #     output = model(image)
    #     # 标签
    #     pre_lab = torch.argmax(output, dim=1)
    #     result = pre_lab.item()
    # print("预测值:", classes[result])
