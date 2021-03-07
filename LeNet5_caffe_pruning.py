import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
from models.LeNet5_caffe import LeNet5_caffe
import copy
from utils.utils import train, get_accuracy, get_sparsity
from utils.prunning import PreTPUnst, PreDNSUnst, Pruned
from utils.NCS import NCS_C

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

#基本参数设置
device = torch.device("cuda")
print(torch.cuda.is_available())
thenet = LeNet5_caffe().to(device=device)
optimizer = optim.Adam(thenet.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

#训练参考模型
# epoch = 10
# path = './weights/best_lenet5_caffe.weight'
# train(epoch, model, device, optimizer, criterion, train_loader, test_loader, path)

#加载训练好的参考模型
weight_dict = torch.load('./weights/best_lenet5_caffe.weight')
thenet.load_state_dict(weight_dict)

#超参数设置
K = 1000
pruning_loops = 15

# 剪枝和调整
loop = 0
best_c = np.zeros(5)

while loop < pruning_loops:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if batch_idx % K == 0:
            if loop < 10:
                ncs = NCS_C(thenet, data, target)
            else:
                ncs = NCS_C(thenet, data, target, sigma=0.5)
            best_c  = ncs.run(thenet, data, target)
            print("best_c = {}".format(best_c))
            PreTPUnst(thenet, best_c)
            Pruned(thenet)
            print("第{}轮剪枝，准确率为{}, 稀疏度为{}".format(loop, get_accuracy(thenet), get_sparsity(thenet)))
            loop = loop + 1
        else:
            optimizer.zero_grad()
            output = thenet(data)
            loss = criterion(output, target)
            loss.backward()
            PreDNSUnst(thenet, best_c)
            optimizer.step()
            Pruned(thenet)

it = 0
iters = 15000
while it < iters:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = thenet(data)
        loss = criterion(output, target)
        loss.backward()
        PreDNSUnst(thenet, best_c)
        optimizer.step()
        Pruned(thenet)
        it = it + 1
        if it % 1000 == 0:
            print("第{}轮恢复，准确率为{}, 稀疏度为{}".format(it, get_accuracy(thenet), get_sparsity(thenet)))

acc = get_accuracy(thenet)
spar = get_sparsity(thenet)
print(acc, spar)
