# 解决相对导入的问题
'''
相对引用，在别的模块调用时不会出错
相对引用，在本模块运行时会报错
解决方法：加入下面的一句话
'''
__package__ = 'Resnet-MNIST'

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from tqdm import tqdm

from .model import *
from .dataloader import train_loader
from .dataloader import test_loader


# # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters
num_epochs = 5

learning_rate = 0.001
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
# model = myNet(784).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # Train the model
    # model.load_state_dict(torch.load('resnet.ckpt'))
    total_step = len(train_loader)
    curr_lr = learning_rate

    for epoch in range(num_epochs):
        # for i, (images, labels) in enumerate(train_loader):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            # images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if (epoch + 1) % 2 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), 'resnet.ckpt')

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)


def test():
    # Test the model
    # model.load_state_dict(torch.load('resnet.ckpt'))
    # model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            # images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # softmax用来多分类。torch.max用来求最大值。1代表行。0的话代表列
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            '''predicted 和 labels都是tensor。(predicted == labels)在两个tensor的对应位置比较，一样为1，不一样为0。
            sum是把对应为位置的数字相加
            item是转换成python的格式'''
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    train()
    test()
