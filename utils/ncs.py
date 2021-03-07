import numpy as np
import copy
from utils.prunning import Apply_TPUnst
from utils.utils import get_sparsity

def evaluate(model, c_set, data, target, acc_orig=0.98, delta = 0.05):
    fitness = np.zeros(c_set.shape[0]) + 1.1
    for i in range(c_set.shape[0]):
        model_copy = copy.deepcopy(model)
        Apply_TPUnst(model_copy, c_set[i])
        output = model_copy(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
        if acc_orig - acc <= delta:
            fitness[i] = get_sparsity(model_copy)
            # print(acc, c_set[i])
            # print(acc)
        # if acc_orig - acc <= delta:
        #     fit = get_sparsity(model) - 1.0
        # else:
        #     fit = (acc_orig - acc) / delta
        # fitness[i] = fit
    return fitness

class NCS_C:

    def __init__(self, model, data, target, Tmax=160, popN=10, sigma=5.0):
        self.Tmax = Tmax
        self.r = 0.95
        self.epoch = 10
        self.popN = popN
        self.bound = [0.0, 20.0]
        self.sigma = np.tile(sigma, (self.popN, 1))
        self.D = 5
        self.x = np.ones((self.popN, self.D)) * 0.001
        # print(self.x)
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
            # print(new_x)

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