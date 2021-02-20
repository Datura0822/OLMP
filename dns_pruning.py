import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.LeNet import LeNet
from models.LeNet5_caffe import LeNet5_caffe
from utils.utils import test, train, sparsity
from utils.DNS import dns_prune
import torch.nn.utils.prune as prune


#下载和加载mnist数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=64, shuffle=True)


#参数配置
device = torch.device("cuda")
print(torch.cuda.is_available())
model = LeNet5_caffe().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
#训练或直接加载训练好的权重
# epoch = 10
# path = './weights/best_lenet5_caffe.weight'
# train(epoch, model, device, optimizer, criterion, train_loader, test_loader, path) #训练
weight_dict = torch.load('./weights/pruned_lenet5_caffe.weight') #直接加载训练好的权重
model.load_state_dict(weight_dict)

#输出原始模型的精确度和参数个数
loss, acc = test(model, criterion, device, test_loader)
param, spar_rate = sparsity(model)
# print(rest, total)
print("Before pruning: the accuracy is {}, the number of param {}({:.2f}%)".format(acc, param, spar_rate*100))

epoch = 100
path = './weights/pruned_lenet5_caffe.weight'
dns_prune(epoch, model, device, optimizer, criterion, train_loader, test_loader, path)






    # # 预剪枝
    # # 重训练
    # path = './weights/final_weight.weight'
    # train(1, model, device, optimizer, criterion, train_loader, test_loader, path)
    #

# from thop import profile
#
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input,))
# print(flops)
# print(params)