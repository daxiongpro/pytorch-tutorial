import torch
import torch.nn as nn
from torch.optim import Adam

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        self.linear2 = nn.Linear(hid,out_dim)
        self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred

in_d = 5
hidn = 2
out_d = 3
net = NN_Network(in_d, hidn, out_d)

for param in net.parameters():
    print(type(param.data), param.size())

print(list(net.parameters()))