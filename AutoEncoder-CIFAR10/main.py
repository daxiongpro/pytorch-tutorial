# 解决相对导入的问题
'''
相对引用，在别的模块调用时不会出错
相对引用，在本模块运行时会报错
解决方法：加入下面的一句话
'''
__package__ = 'AutoEncoder-CIFAR10'

import cv2
import torchvision
from tqdm import tqdm
from .model import *
from .dataloader import train_loader
from .dataloader import test_loader


# # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 30
learning_rate = 0.001

model = AutoEncoderNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    # Train the model
    model.load_state_dict(torch.load('AENet-CIFAR10.ckpt'))
    for epoch in range(num_epochs):
        loss_total = 0
        print('\nepoch {}'.format(epoch + 1))
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print('loss: {}'.format(loss.item()))

        if (epoch + 1) % 1 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), 'AENet-CIFAR10.ckpt')

def test():
    # Test the model
    model.load_state_dict(torch.load('AENet-CIFAR10.ckpt'))
    model.eval()
    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).cpu()

            # 重建
            # 原始数据
            img = torchvision.utils.make_grid(images.cpu(), nrow=10)
            img = img.numpy().transpose(1, 2, 0)
            # 经过autoencoder之后的数据
            reimg = torchvision.utils.make_grid(outputs, nrow=10)
            reimg = reimg.numpy().transpose(1, 2, 0)

        cv2.imshow('img', img)
        cv2.imshow('reimg', reimg)
        cv2.waitKey(0)

if __name__ == '__main__':
    train()
    test()
