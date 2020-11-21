import torch
import torchvision
import torch.utils.data as Data
import scipy.misc
import os
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable

BATCH_SIZE = 100
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(root='../data/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)

test_data = torchvision.datasets.MNIST(root='../data/',
                                       train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# 具体查看图像形式为：

images, lables = next(iter(train_loader))

# nrow：每行显示多少个数字
img = torchvision.utils.make_grid(images, nrow = 20)
img = img.numpy().transpose(1, 2, 0)
cv2.imshow('img', img)
cv2.waitKey(0)