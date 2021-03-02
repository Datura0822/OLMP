import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
from models.LeNet import LeNet
from models.LeNet5_caffe import LeNet5_caffe
import copy

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
#
#加载训练好的参考模型
weight_dict = torch.load('./weights/best_lenet5_caffe.weight')
thenet.load_state_dict(weight_dict)

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
#
# print(get_accuracy(model))
def get_sparsity(model):
    rest = sum([torch.sum(param != 0) for param in model.parameters()])
    total = sum([param.numel() for param in model.parameters()])
    return rest / total
#
def evaluate(model, c_set, data, target, acc_orig=0.99, delta = 0.05):
    fitness = np.zeros(c_set.shape[0]) + 1.1
    for i in range(c_set.shape[0]):
        model_copy = copy.deepcopy(model)
        # print(get_sparsity(model_copy))
        apply_prune(model_copy, c_set[i])
        output = model_copy(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
        if acc_orig - acc <= delta:
            fitness[i] = get_sparsity(model_copy)
        # if acc_orig - acc <= delta:
        #     fit = get_sparsity(model) - 1.0
        # else:
        #     fit = (acc_orig - acc) / delta
        # fitness[i] = fit
    return fitness
#
class Threshold_Pruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, c):
        self.c = c

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[torch.where(t.abs() < self.c)] = 0
        return mask

def tp_unstructured(module, name, c):
    Threshold_Pruning.apply(module, name, c)
    return module

def apply_prune(model, c):
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            tp_unstructured(module, name='weight', c=c[i])
            tp_unstructured(module, name='bias', c=c[i])
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        elif isinstance(module, torch.nn.Linear):
            tp_unstructured(module, name='weight', c=c[i])
            tp_unstructured(module, name='bias', c=c[i])
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        i = i + 1
#

# 测试
# for data, target in test_loader:
#     c_set = np.ones((5, 4))
#     apply_prune(model, c_set[0])
#     break
class DynamicNetworkSurgery(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, c):
        # Check range of validity of pruning amount
        self.c = 1

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        t_mean = t.mean()
        t_std = t.std()
        a = 0.9 * max(t_mean + self.c * t_std, 0)
        b = 1.1 * max(t_mean + self.c * t_std, 0)
        mask[torch.where(t.abs() < a)] = 0
        mask[torch.where(t.abs() > b)] = 1
        return mask

def dns_unstructured(module, name, c):
    DynamicNetworkSurgery.apply(module, name, c)
    return module


#动态网络手术
def pre_prune(model, c):
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            dns_unstructured(module, name='weight', c=c[i])
            dns_unstructured(module, name='bias', c=c[i])

        elif isinstance(module, torch.nn.Linear):
            dns_unstructured(module, name='weight', c=c[i])
            dns_unstructured(module, name='bias', c=c[i])

        i = i + 1

def pruned(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

class NCS_C:
    'This class contain the alogirhtm of NCS, and its API for invoking.'
    def __init__(self, model, data, target, Tmax=160, popN=10, sigma=5.0):
        self.Tmax = Tmax
        self.r = 0.95
        self.epoch = 10
        self.popN = popN
        self.bound = [0.0, 20.0]
        self.sigma = np.tile(sigma, (self.popN, 1))
        self.D = 5
        self.x = np.ones((self.popN, self.D)) * 0.001
        # print(get_sparsity(model))
        self.fit = evaluate(model, self.x, data, target)
        pos = np.argmin(self.fit)
        self.bestfound_x = self.x[pos]
        self.bestfound_fit = self.fit[pos]

    def Corr(self, x, new_x):
        Corrp = 1e300*np.ones(self.popN)
        new_Corrp = 1e300 * np.ones(self.popN)
        for i in range(self.popN):
            Db = 1e300*np.ones(self.popN)
            new_Db = 1e300 * np.ones(self.popN)
            for j in range(self.popN):
                if i != j:
                    sigma = (self.sigma[i]**2 + self.sigma[j]**2) / 2
                    E_inv = np.identity(self.D) / sigma
                    tmp =  (1 / 2) * self.D * (np.log(sigma) - np.log(self.sigma[i]*self.sigma[j]))
                    Db[j] = (1 / 8) * (x[i] - x[j]) @ E_inv @ (x[i] - x[j]).T + tmp
                    new_Db[j] = (1 / 8) * (new_x[i] - x[j]) @ E_inv @ (new_x[i] - x[j]).T + tmp
            Corrp[i] = np.min(Db)
            new_Corrp[i] = np.min(new_Db)
        return Corrp, new_Corrp

    def run(self, model, data, target):
        t = 0
        c = np.zeros((self.popN, 1))
        while t < self.Tmax:
            # 更新lambda t
            self.lambdat = 1.0 + np.random.randn(1) * (0.1 - 0.1 * t / self.Tmax)

            # 产生新种群x'
            new_x = self.x + self.sigma * np.random.randn(self.popN, self.D)

            # 检查边界
            pos = np.where(new_x < self.bound[0])
            new_x[pos] = self.bound[0] + 0.0001
            pos = np.where(new_x > self.bound[1])
            new_x[pos] = self.bound[1] - 0.0001

            # 计算 f(x'),
            new_fit = evaluate(model, new_x, data, target)
            # 计算 Corr(p)和Corr(p')
            Corrp, new_Corrp = self.Corr(self.x, new_x)

            # 更新 BestFound
            pos = np.argmin(self.fit)
            if new_fit[pos] < self.bestfound_fit:
                self.bestfound_x = new_x[pos]
                self.bestfound_fit = new_fit[pos]

            # the normalization step
            norm_fit = (new_fit - self.bestfound_fit)  / (self.fit + new_fit - 2 * self.bestfound_fit)
            norm_Corrp = new_Corrp / (Corrp + new_Corrp)
            # 更新 x
            pos = np.where(norm_fit  < self.lambdat * norm_Corrp)
            self.x[pos] = new_x[pos]
            self.fit[pos] = new_fit[pos]
            c[pos] += 1
            t += 1
            #1/5 successful rule
            if t % self.epoch == 0:
                for i in range(self.popN):
                    if c[i][0] > 0.2 * self.epoch:
                        self.sigma[i][0] /= self.r
                    elif c[i][0] < 0.2 * self.epoch:
                        self.sigma[i][0] *= self.r
                c = np.zeros((self.popN, 1))
                # print('the {} {}'.format(t, self.bestfound_fit))
        return self.bestfound_x

#
#超参数设置
K = 1000
pruning_loops = 15
#
# #剪枝和调整
loop = 0
best_c = np.zeros(5)

while loop < pruning_loops:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if batch_idx % K == 0:
            print("第{}轮剪枝，准确率为{}, 稀疏度为{}".format(loop, get_accuracy(thenet), get_sparsity(thenet)))
            ncs = NCS_C(thenet, data, target)
            best_c = ncs.run(thenet, data, target)
            print("best_c = {}".format(best_c))
            apply_prune(thenet, best_c)
            loop = loop + 1
        else:
            optimizer.zero_grad()
            output = thenet(data)
            loss = criterion(output, target)
            loss.backward()
            pre_prune(thenet, best_c)
            optimizer.step()
            pruned(thenet)

it = 0
iters = 15000
while it < iters:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = thenet(data)
        loss = criterion(output, target)
        loss.backward()
        pre_prune(thenet, best_c)
        optimizer.step()
        pruned(thenet)
        it = it + 1
        if it % 1000 == 0:
            print("第{}轮恢复，准确率为{}, 稀疏度为{}".format(it, get_accuracy(thenet), get_sparsity(thenet)))

acc = get_accuracy(thenet)
spar = get_sparsity(thenet)
print(acc, spar)
