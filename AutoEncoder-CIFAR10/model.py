'''
卷积层模板不要太多。output channel少一点，有利于提升速度和精度
全连接层就一层够了，之前弄了多层，效果不好，训练速度也很慢
'''
import torch.nn as nn

class AutoEncoderNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.fc4 = nn.Linear(10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 32 * 32)
        self.fc7 = nn.Linear(32 * 32, 32 * 32 * 3)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        # self.bn = nn.BatchNorm1d()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # 把数据变成(batch_size, 3*32*32)
        out = out.view(out.size(0), -1)

        out = self.fc7(out)
        out = out.reshape(100, 3, 32, 32)
        return out
