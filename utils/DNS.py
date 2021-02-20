import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from models.LeNet import LeNet
from torchvision import datasets, transforms
from utils.utils import test, sparsity

class DynamicNetworkSurgery(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self):
        # Check range of validity of pruning amount
        self.c = 1.0

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        t_mean = t.mean()
        t_std = t.std()
        a = 0.9 * max(t_mean + self.c * t_std, 0)
        b = 1.1 * max(t_mean + self.c * t_std, 0)
        # print("t={}".format(t))
        # print("a={}, b={}".format(a,b))
        # print(t.abs() < a)
        mask[torch.where(t.abs()<a)] = 0
        mask[torch.where(t.abs()>b)] = 1

        # print("mask={}".format(mask))
        # print(mask*t)
        return mask

def dns_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        m = nn.Linear(3, 4)
        foobar_unstructured(m, name='bias')
    """
    DynamicNetworkSurgery.apply(module, name)
    return module


#动态网络手术
def pre_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            dns_unstructured(module, name='weight')
            dns_unstructured(module, name='bias')

        elif isinstance(module, torch.nn.Linear):
            dns_unstructured(module, name='weight')
            dns_unstructured(module, name='bias')

def pruned(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')


def dns_prune(epoch, model, device, optimizer, criterion, train_loader, test_loader, path):
    _, init_acc = test(model, criterion, device, test_loader)
    best_epoch = -1
    best_acc = -1
    best_spar_rate = 1.0
    for iter in range(1, epoch + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device=device), target.to(device=device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            pre_prune(model)
            optimizer.step()
            pruned(model)

            if batch_idx % 100 == 0:
                print('Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

        # 输出剪枝后模型的精确度
        loss, acc = test(model, criterion, device, test_loader)
        param, spar_rate = sparsity(model)
        if acc - init_acc > -0.008 and spar_rate < best_spar_rate:
            best_spar_rate = spar_rate
            best_epoch = iter
            best_acc = acc
            torch.save(model.state_dict(), path)

        print("After {} epoch pruning: the accuracy is {}, the number of param {}({:.2f}%)".format(iter, acc, param, spar_rate*100))

    print("Finally, the best epoch is {} with the accuracy is {} and the compression ratio is {:.2f}%".format(best_epoch, best_acc, best_spar_rate*100))

# def adjust_learning_rate(lr_0, lr_n, iter, n):
#     lr = lr_0 * (1 - iter / n) + lr_n * iter / n


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LeNet().to(device=device)
# weight_dict = torch.load('./weights/best_model.weight') #加载权重
# print(weight_dict)
# model_dict = model.state_dict()
# weight_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
# model_dict.update(weight_dict)
# model.load_state_dict(weight_dict)
# print(list(model.parameters()))


# for name, module in model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         dns_unstructured(module, name='bias', amount=0.2)
#         # 将所有卷积层的权重减去 20%
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         dns_unstructured(module, name='bias', amount=0.4)
#         # 将所有全连接层的权重减去 40%

# print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist
# dns_unstructured(model.fc3, name='bias', amount=0.3)
# #
# print(model.state_dict())

