from torch import nn
import torch.nn.functional as F

class LeNet300_100(nn.Module):
    def __init__(self):
        super(LeNet300_100, self).__init__()

        self.fc1 = nn.Linear(28 * 28 * 1, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
