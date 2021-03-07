import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, c):
        self.c = c

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        t_mean = t.mean()
        t_std = t.std()
        a = 0.9 * max(t_mean + self.c * t_std, 0)
        mask[torch.where(t.abs() < a)] = 0
        return mask

def TPUnst(module, name, c):
    # 无结构阈值剪枝
    ThresholdPruning.apply(module, name, c)
    return module


def PreTPUnst(model, c):
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            TPUnst(module, name='weight', c=c[i])
            TPUnst(module, name='bias', c=c[i])

        elif isinstance(module, torch.nn.Linear):
            TPUnst(module, name='weight', c=c[i])
            TPUnst(module, name='bias', c=c[i])

        i = i + 1

def Apply_TPUnst(model, c):
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            TPUnst(module, name='weight', c=c[i])
            TPUnst(module, name='bias', c=c[i])
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        elif isinstance(module, torch.nn.Linear):
            TPUnst(module, name='weight', c=c[i])
            TPUnst(module, name='bias', c=c[i])
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        i = i + 1


class DynamicNetworkSurgery(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, c):
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


def DNSUnst(module, name, c):
    #  无结构DNS剪枝
    DynamicNetworkSurgery.apply(module, name, c)
    return module


# 动态网络手术
# def Apply_DNSUnst(model, c):
#     i = 0
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d):
#             DNSUnst(module, name='weight', c=c[i])
#             DNSUnst(module, name='bias', c=c[i])
#             prune.remove(module, 'weight')
#             prune.remove(module, 'bias')
#
#         elif isinstance(module, torch.nn.Linear):
#             DNSUnst(module, name='weight', c=c[i])
#             DNSUnst(module, name='bias', c=c[i])
#             prune.remove(module, 'weight')
#             prune.remove(module, 'bias')
#
#         i = i + 1
def PreDNSUnst(model, c):
    i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            DNSUnst(module, name='weight', c=c[i])
            DNSUnst(module, name='bias', c=c[i])

        elif isinstance(module, torch.nn.Linear):
            DNSUnst(module, name='weight', c=c[i])
            DNSUnst(module, name='bias', c=c[i])

        i = i + 1


def Pruned(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')

        elif isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')
