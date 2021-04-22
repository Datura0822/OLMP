import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.LeNet5_caffe import LeNet5_caffe
from utils.utils import get_accuracy, get_sparsity
from utils.prunning import PreTPUnst, PreDNSUnst, Pruned
from utils.ncs import NCS_C
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"


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
# print(torch.cuda.is_available())
thenet = LeNet5_caffe().to(device=device)

#加载训练好的参考模型
weight_dict = torch.load('./weights/best_lenet5_caffe.weight')
thenet.load_state_dict(weight_dict)
print("原始准确率为{}, 原始稀疏度为{}".format(get_accuracy(thenet), get_sparsity(thenet)))

#超参数设置
K = 1000
pruning_loops = 15

# 剪枝和调整
best_c = np.zeros(5)
it = 1

while it <= 30000:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer = torch.optim.SGD(thenet.parameters(), lr=0.01*(1+0.0001*it)**(-0.75), momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()
        output = thenet(data)
        loss = criterion(output, target)
        loss.backward()
        if it <= 15000 and it % 1000 == 1:
            # 剪枝
            if it / K <= 10.0:
                ncs = NCS_C(thenet, data, target)
            else:
                ncs = NCS_C(thenet, data, target, sigma=0.5)
            best_c  = ncs.run(thenet, data, target)
            PreTPUnst(thenet, best_c)
        elif it <= 15000:
            # 恢复
            PreDNSUnst(thenet, best_c)
        else:
            # 调整
            PreDNSUnst(thenet, best_c)
        optimizer.step()
        Pruned(thenet)
        if it % 1000 == 0:
            print("第{}轮，准确率为{}, 稀疏度为{}".format(it, get_accuracy(thenet), get_sparsity(thenet)))
        it = it + 1

top1_error = 1.0 - get_accuracy(thenet)
pr = 1.0 / get_sparsity(thenet)
print(top1_error, pr)
torch.save(thenet.state_dict(), './weights/pruned_lenet5_caffe.weight')
