# CIFAR-10 dataset
import torch
import torchvision
# Image preprocessing modules
from torchvision import transforms
import torchvision.transforms as transforms

batch_size = 100

transform = transforms.Compose([
    # transforms.Resize([784, 1]),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root='../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                            train=False,
                                            transform=transform)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)