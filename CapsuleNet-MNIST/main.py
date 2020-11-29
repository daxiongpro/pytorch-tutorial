
__package__ = 'CapsuleNet-MNIST'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from .model import CapsNet
from .data_loader import Dataset
from tqdm import tqdm

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_capsule_length = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_capsule_num = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_out_caps_num = 10
            self.dc_in_caps_num = 32 * 6 * 6
            self.dc_in_caps_length = 8
            self.dc_out_caps_length = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'your own dataset':
            pass


def train(model, optimizer, train_loader, epoch):
    capsule_net = model
    capsule_net.train()  # 将本层及子层的training设定为True
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = capsule_net(data)
        loss = capsule_net.margin_loss(output, target)
        loss.backward()
        optimizer.step()
        # correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        train_loss = loss.item()
        total_loss += train_loss
        if (epoch + 1) % 1 == 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), './CapsNet.ckpt')

        if batch_id % 100 == 0:
            # tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                # correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch, N_EPOCHS, total_loss / len(train_loader.dataset)))


def test(capsule_net, test_loader, epoch):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output = capsule_net(data)
        loss = capsule_net.margin_loss(output, target)

        test_loss += loss.item()

    tqdm.write("Epoch: [{}/{}], loss: {:.6f}".format(epoch, N_EPOCHS, test_loss / len(test_loader)))


if __name__ == '__main__':
    torch.manual_seed(1)
    # dataset = 'cifar10'
    dataset = 'mnist'
    config = Config(dataset)
    mnist = Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet()
    # 使用多个GPU进行训练
    capsule_net = torch.nn.DataParallel(capsule_net, device_ids=[0])
    if USE_CUDA:
        capsule_net = capsule_net.cuda()

    '''
    调用model里的函数 继承的函数可以直接调用 
    例如 model.state_dict() ,model.load_state_dict(torch.load(model_path)......不受影响。
    但是自己写的函数 要加上.module才行  model.module.forward_getfeature(x)。
    自己写的函数 不可以并行运算 ，只能在主gpu中运算。
    '''
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters())

    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e)
        test(capsule_net, mnist.test_loader, e)
