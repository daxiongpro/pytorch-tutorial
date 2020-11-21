import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AutoEncoderNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)

        self.fc6 = nn.Linear(512, 784)
        self.fc5 = nn.Linear(100, 512)
        self.fc4 = nn.Linear(10, 100)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm1d()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # out = self.fc3(out)
        # out = self.relu(out)
        # out = self.fc4(out)
        # out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = out.reshape(100, 1, 28, 28)
        return out

class AutoEncoderNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)

        self.fc2 = nn.Linear(100, 784)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm1d()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.reshape(100, 1, 28, 28)
        return out