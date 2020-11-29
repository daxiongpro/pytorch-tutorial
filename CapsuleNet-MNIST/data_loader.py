import torch
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
        if dataset == 'mnist':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST('../data/', train=True, download=True,
                                           transform=dataset_transform)
            test_dataset = datasets.MNIST('../data/', train=False, download=True,
                                          transform=dataset_transform)

            # num_workers = 2:使用2个线程读取数据
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                '../data/', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                '../data/', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        elif dataset == 'office-caltech':
            pass
        elif dataset == 'office31':
            pass
