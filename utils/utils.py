import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.LeNet import LeNet
import os

device = torch.device("cuda")
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=32, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=64, shuffle=True)

def train(epoch, model, optimizer, criterion, path):
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

        acc = get_accuracy(model)
        print("Epoch {}: accuracy {}".format(iter, acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = iter
            torch.save(model.state_dict(), path)

    print("Best epoch {} with accuracy {:.3f}".format(best_epoch, best_acc))

def get_accuracy(model):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device), target.to(device=device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

def get_sparsity(model):
    rest = sum([torch.sum(param != 0) for param in model.parameters()])
    total = sum([param.numel() for param in model.parameters()])
    return rest / total

# def test(model, criterion, device, test_loader):
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device=device), target.to(device=device)
#             output = model(data)
#             test_loss += criterion(output, target).item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     correct /= len(test_loader.dataset)
#
#     # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
#     #     test_loss, correct, len(test_loader.dataset),
#     #     100. * correct / len(test_loader.dataset)))
#     return test_loss, correct