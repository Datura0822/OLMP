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