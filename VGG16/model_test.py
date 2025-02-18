# @Author : LiZhongzheng
# 开发时间  ：2025-02-14 9:22
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import VGG16


def test_data_process():
    # 使用验证集
    test_data = FashionMNIST(root='./data', train=False,
                             transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                             download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=2)

    return test_loader


test_loader = test_data_process()


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 初始化参数
    # test_correct = 0.0只有改为张量形式下面才可以使用.double()方法
    test_num = 0

    test_correct = torch.tensor(0.0, dtype=torch.float32)  # 初始化为张量

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            # 找到张量（Tensor）中最大值的索引
            pre_lab = torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y.data).item()
            # 将所有的测试样本累加
            test_num += test_data_x.size(0)
    # 计算准确率
    test_acc = test_correct.double().item() / test_num
    print("测试的准确率：", test_acc)


if __name__ == '__main__':
    model = VGG16()
    # 加载模型权重
    model.load_state_dict(torch.load("best_model.pth"))
    test_loader = test_data_process()
    test_model_process(model, test_loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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
