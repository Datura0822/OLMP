import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.LeNet import LeNet
from models.LeNet5_caffe import LeNet5_caffe

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
model = LeNet5_caffe().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

net = {
    'device':device,
    'model':model,
    'optimizer':optimizer,
    'criterion':criterion,
    'train_loader':train_loader,
    'test_loader':test_loader
}

#超参数设置
param = {'K':1000,
         'pruning_loops':15,
         'popN': 10,
         'Tmax':160,
         'delta':0.005,
         'sigma':5.0
}

# K = 1000
# pruning_loops = 15
# popN = 10
# Tmax = 160
# delta = 0.005
# sigma = 5.0

#训练参考模型
# epoch = 10
# path = './weights/best_lenet5_caffe.weight'
# train(epoch, model, device, optimizer, criterion, train_loader, test_loader, path)

#加载训练好的参考模型
weight_dict = torch.load('./weights/best_lenet5_caffe.weight')
model.load_state_dict(weight_dict)

#剪枝和调整
for loop in range(pruning_loops):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if batch_idx == 0:
            ncs()
        elif batch_idx < K:


        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



print(param['Tmax'])
# kk = iter(enumerate(train_loader))
# print(next(kk))
# print(enumerate(train_loader).size())