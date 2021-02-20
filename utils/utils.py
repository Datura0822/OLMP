import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.LeNet import LeNet
import os

# def train(model, device, train_loader, optimizer, criterion):
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#
#         if batch_idx % 100 == 0:
#             print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))

class trainer:
    def __init__(self, args):
        self.model = model  # 构建模型
        if args.cuda:
            self.model = self.model.cuda()  # 如果有可用的gpu,在构建模型的优化器之前要先把模型导入到gpu之中
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=args.lr)  # 构建一个SDG类的实例
        self.loss = torch.nn.nn.CrossEntropyLoss()  # 构建一个损失函数类实例
        if args.cuda:
            self.loss = self.loss.cuda()  # 同理导入近gpu

    def traing(self):


def test(model, criterion, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device), target.to(device=device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return test_loss, correct


def sparsity(model):
    rest = sum([torch.sum(param != 0) for param in model.parameters()])
    total = sum([param.numel() for param in model.parameters()])
    return rest, rest / total

def train(epoch, model, device, optimizer, criterion, train_loader, test_loader, path):
    best_acc = -1
    best_epoch = -1
    for iter in range(1, epoch + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        loss, acc = test(model, criterion, device, test_loader)
        print("Epoch {}: accuracy {}".format(iter, acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = iter
            torch.save(model.state_dict(), path)

    print("Best epoch {} with accuracy {:.3f}".format(best_epoch, best_acc))


