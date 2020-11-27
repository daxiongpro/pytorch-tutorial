import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    """
    输入:(B,C,H,W)=(B,1,28,28)
    输出：(B,C,H,W)=(B,256,20,20)
    """

    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    """
        输入:(B,C,H,W)=(B,256,20,20)
        输出：(B,C_N,C_L)=(B,32*6*6, 8)=(B,1152,8)
        C_N:capsule_num，胶囊的个数
        C_L:capsule_length，每个胶囊的长度
    """

    def __init__(self, capsule_length=8, in_channels=256, out_channels=32, capsule_num=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.capsule_length = capsule_length
        self.capsule_num = capsule_num
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * capsule_length,
                              kernel_size=9,
                              stride=2,
                              padding=0)

    def forward(self, x):
        """
        :param x: (B,C,H,W) -> (B,256,20,20)
        :return: (B,C_N,C_L) -> (100,32*6*6,8) = (100,1152,8)
        """
        x = self.conv(x)  # (B,256,6,6)
        x = self.toCapsules(x)  # (B,32*6*6, 8) =(B,1152,8)
        return x

    def toCapsules(self, x):
        # x:(B,256,6,6)
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        x = x.reshape(B, self.capsule_length, -1, H, W)
        x = x.reshape(B, self.capsule_length, -1)
        x = x.permute(0, 2, 1)  # (B, 32*6*6, 8)
        return x


class DigitCaps(nn.Module):
    def __init__(self, in_capsule_num=32 * 6 * 6, out_capsule_num=10, in_cap_length=8, out_cap_length=16):
        super(DigitCaps, self).__init__()

        self.in_capsule_num = in_capsule_num
        self.out_capsule_num = out_capsule_num
        self.in_cap_length = in_cap_length
        self.out_cap_length = out_cap_length

        self.W = nn.Parameter(torch.randn(in_capsule_num, out_capsule_num, out_cap_length, in_cap_length)).cuda()

    def forward(self, x):
        '''
        :param x: torch.Size([B, 1152, 8])
        :return: (B, 10, 16)
        '''

        B = x.size(0)
        # unsqueeze(2)：在第4个维度上增加一个括号，即若原来是(2,3,3,5), 将变成(2,3,1,3,5)
        x = x.unsqueeze(2)  # torch.Size([B, 1152, 1, 8]
        x = x.expand(B, 1152, 10, 8)  # torch.Size([B, 1152, 10, 8]
        x = x.unsqueeze(4)  # torch.Size([B, 1152, 10, 8, 1]

        W = self.W.unsqueeze(0)  # [1, 1152, 10, 16, 8]
        W = W.expand(B, self.in_capsule_num, self.out_capsule_num, self.out_cap_length, self.in_cap_length)  # W：torch.Size([B, 1152, 10, 16, 8])

        u_hat = torch.matmul(W, x)  # u_hat:torch.Size([B, 1152, 10, 16, 1])
        u_hat = u_hat.squeeze(-1)  # u_hat:torch.Size([B, 1152, 10, 16])

        b_ij = Variable(torch.zeros(self.in_capsule_num, self.out_capsule_num))  # (1152, 10)
        b_ij = b_ij.unsqueeze(0).expand(B, self.in_capsule_num, self.out_capsule_num)  # (B, 1152, 10)
        b_ij = b_ij.unsqueeze(3)  # (B, 1152, 10, 1)

        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(1, num_iterations + 1):
            c_ij = F.softmax(b_ij, dim=1)  # (B, 1152, 10, 1)
            c_ij = c_ij.expand(B, self.in_capsule_num, self.out_capsule_num, self.out_cap_length)  # (B, 1152, 10, 16)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # (B, 1, 10, 16)
            v_j = self.squash(s_j)  # (B, 1, 10, 16) 16是经过squash之后的，其余维度都为原来的平方？

            if iteration < num_iterations:  # 最后一步不需要下面两行。节约计算
                a_ij = u_hat * v_j.expand(B, self.in_capsule_num, self.out_capsule_num, self.out_cap_length).sum(-1, keepdim=True)  # (B, 1152, 10, 1)
                b_ij = b_ij + a_ij

        return v_j.squeeze(1)  # (B, 10, 16)

    def squash(self, input_tensor):
        """
        input_tensor: (B, 1, 10, 16)
        return: output_tensor: (B, 1, 10, 16)
        """
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)  # (B, 1, 10, 1)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        # reconstructions, masked = self.decoder(output, data)
        # return output, reconstructions, masked
        return output

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        # x: (B, 10, 16)
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))  # (B, 10, 1)


        left = (0.9 - v_c).view(batch_size, -1)  # [100, 10]
        right = (v_c - 0.1).view(batch_size, -1) # [100, 10]

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005


if __name__ == '__main__':
    net = CapsNet(config=None).cuda()

    input = torch.rand(100, 10, 16).cuda()
    labels = torch.rand(100, 10).cuda()

    input = torch.rand(100, 1, 28, 28).cuda()
    output = net(input)
    print(output.shape)
    # layer.margin_loss(input, labels)

