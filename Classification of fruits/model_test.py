# @Author : LiZhongzheng
# 开发时间  ：2025-02-14 9:22
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception
from torchvision.datasets import ImageFolder
from PIL import Image


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'data\test'

    normalize = transforms.Normalize([0.22890568, 0.19639583, 0.1433638], [0.09928422, 0.08263004, 0.06472758])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == '__main__':
    model = GoogLeNet(Inception)
    # 加载模型权重
    model.load_state_dict(torch.load("./best_model.pth"))

    ###############################################################################################
    test_loader = test_data_process()
    test_model_process(model, test_loader)
    ###############################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 注意：要按照文件的排序去写
    classes = ["苹果", "香蕉", '葡萄', '橙子/橘子', '梨']

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
    # normalize = transforms.Normalize([0.22890568, 0.19639583, 0.1433638], [0.09950783, 0.07997292, 0.06596899])
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
