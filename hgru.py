import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from scipy import ndimage as ndi

import cv2
import scipy
from torch.nn import init


class FFConvNet(nn.Module):

    def __init__(self,in_shape):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, bias=False)
        kernel = np.load("gabor_serre.npy")
        self.conv0.weight.data = torch.FloatTensor(kernel)
        self.conv1 = nn.Conv2d(25, 9, kernel_size=20)
        self.conv2 = nn.Conv2d(9, 9, kernel_size=20)
        self.conv3 = nn.Conv2d(9, 9, kernel_size=20)
        self.conv4 = nn.Conv2d(9, 9, kernel_size=20)
        self.conv5 = nn.Conv2d(9, 9, kernel_size=20)
        self.conv6 = nn.Conv2d(9,2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.bn
        self.bn1 = nn.BatchNorm2d(9)
        self.bn2 = nn.BatchNorm2d(9)
        self.bn3 = nn.BatchNorm2d(9)
        self.bn4 = nn.BatchNorm2d(9)
        self.bn5 = nn.BatchNorm2d(9)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        #self.dropout4 = torch.nn.Dropout(p=0.2)
        self.avgpool = nn.MaxPool2d(155, stride=1)
        self.fc = nn.Linear(6, 6)


    def forward(self, x):
        x = self.conv0(x)
        x = torch.pow(x, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x_n.size(0), -1)
        x = self.fc(x)

        return x

class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.w_gate = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, kernel_size, kernel_size))
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.alpha = torch.nn.Parameter(torch.randn(hidden_size,1,1))
        self.gamma = torch.nn.Parameter(torch.randn(hidden_size,1,1))
        self.kappa = torch.nn.Parameter(torch.randn(hidden_size,1,1))
        self.w = torch.nn.Parameter(torch.randn(hidden_size,1,1))
        self.mu= torch.nn.Parameter(torch.randn(hidden_size,1,1))

        if self.batchnorm:
            self.bn = nn.ModuleList([BatchNorm2d1(25, momentum=0.001, eps=1e-03) for i in range(self.timesteps*4)])
        else:
            self.n = torch.nn.Parameter(torch.randn(self.timesteps,1,1))


        init.xavier_normal_(self.w_gate)
        init.xavier_normal_(self.u1_gate.weight)
        init.xavier_normal_(self.u2_gate.weight)
        self.u1_gate.bias.data =  torch.log(torch.nn.init.uniform(self.u1_gate.bias.data, 1, 8.0 - 1))
        self.u2_gate.bias.data =  torch.log(torch.nn.init.uniform(self.u2_gate.bias.data, 1, 8.0 - 1))
        init.xavier_normal_(self.alpha)
        init.xavier_normal_(self.gamma)
        init.xavier_normal_(self.mu)
        init.xavier_normal_(self.kappa)
        init.xavier_normal_(self.w)

    def forward(self, input_, prev_state2, timestep=0):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if timestep == 0:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state2 = torch.randn(state_size)
            init.xavier_normal_(prev_state2)
            prev_state2 = torch.autograd.Variable(prev_state2).cuda()

        # data size is [batch, channel, height, width]
        kernel = (self.w_gate + self.w_gate.permute(1,0,2,3)) * 0.5
        if self.batchnorm:

            g1_t = torch.nn.functional.sigmoid(self.bn[i*4](self.u1_gate(prev_state2)))
            c1_t = self.bn[i*4+1](torch.nn.functional.conv2d(prev_state2 * g1_t, kernel, padding=self.padding))
            next_state1 = torch.nn.functional.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = torch.nn.functional.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = self.bn[i*4+3](torch.nn.functional.conv2d(next_state1, kernel, padding=self.padding))
            h2_t = torch.nn.functional.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            prev_state2 = ((1 - g2_t)*prev_state2 + g2_t*h2_t)

        else:

            g1_t = torch.nn.functional.sigmoid(self.u1_gate(prev_state2))
            c1_t = torch.nn.functional.conv2d(prev_state2 * g1_t, kernel, padding=self.padding)
            next_state1 = torch.nn.functional.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = torch.nn.functional.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = torch.nn.functional.conv2d(next_state1, kernel, padding=self.padding)
            h2_t = torch.nn.functional.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            prev_state2 = self.n[timestep,...]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        return prev_state2


class hConvGRU(nn.Module):

    def __init__(self, timesteps=8):
        super().__init__()

        self.conv0 = nn.Conv2d(1, 25, kernel_size=7)
        self.timesteps = timesteps
        part1 = np.load("gabor_serre.npy")
        self.conv0.weight.data = torch.FloatTensor(part1)
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        torch.nn.init.xavier_uniform(self.conv6.weight.data)
        self.bn2 = nn.BatchNorm2d(2)
        self.bn = nn.BatchNorm2d(2)
        self.maxpool = nn.MaxPool2d(144, stride=1)
        self.fc = nn.Linear(2, 2)

        self.unit1 = hConvGRUCell3(25, 25, 3)
        self.unit1.train()


    def forward(self, x):
        internal_state = None
        x = x
        x = self.conv0(x)
        x = torch.pow(x, 2)

        for i in range(self.timesteps):
            internal_state  = self.unit1(x, internal_state, timestep=i)

        output = self.conv6(internal_state)
        output = self.bn(output)
        output = self.maxpool(output)
        output = self.bn2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
