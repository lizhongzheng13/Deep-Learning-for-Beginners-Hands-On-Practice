# @Author : LiZhongzheng
# 开发时间  ：2025-02-10 9:48

# 环境测试：测试显卡类型等操作
import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

# import torch

cuda_version = torch.version.cuda
print("Cuda Version:", cuda_version)

cudnn_version = torch.backends.cudnn.version()
print("Cudnn Version:", cudnn_version)
