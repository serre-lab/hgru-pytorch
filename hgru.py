import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

#torch.manual_seed(42)

class FFConvNet(nn.Module):

    def __init__(self, timesteps=8):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, bias=False, padding=3)
        kernel = np.load("gabor_serre.npy")
        self.conv0.weight.data = torch.FloatTensor(kernel)
        
        self.conv1 = nn.Conv2d(25, 9, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm2d(9)
        
        self.conv2 = nn.Conv2d(9, 9, kernel_size=19, padding=9)
        self.bn2 = nn.BatchNorm2d(9)
        
        self.conv3 = nn.Conv2d(9, 9, kernel_size=19, padding=9)
        self.bn3 = nn.BatchNorm2d(9)
        
        self.conv4 = nn.Conv2d(9, 9, kernel_size=19, padding=9)
        self.bn4 = nn.BatchNorm2d(9)
        
        self.conv5 = nn.Conv2d(9, 9, kernel_size=19, padding=9)
        self.bn5 = nn.BatchNorm2d(9)
        
        self.conv6 = nn.Conv2d(9,2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

        #self.dropout4 = torch.nn.Dropout(p=0.2)
        self.avgpool = nn.MaxPool2d(150, stride=1)
        self.fc = nn.Linear(2, 2)


    def forward(self, x):
        x = self.conv0(x)
        x = torch.pow(x, 2)
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv6(x)
        x = self.bn(x)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
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
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.batchnorm:
            #self.bn = nn.ModuleList([nn.GroupNorm(25, 25, eps=1e-03) for i in range(32)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(25, eps=1e-03) for i in range(32)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
#        self.w_gate_inh = nn.Parameter(self.w_gate_inh.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
#        self.w_gate_exc = nn.Parameter(self.w_gate_exc.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
#        self.w_gate_inh.register_hook(lambda grad: print("inh"))
#        self.w_gate_exc.register_hook(lambda grad: print("exc"))
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data


    def forward(self, input_, prev_state2, timestep=0):

        if timestep == 0:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)

        #import pdb; pdb.set_trace()
        i = timestep
        if self.batchnorm:
            g1_t = torch.sigmoid(self.bn[i*4+0](self.u1_gate(prev_state2)))
            c1_t = self.bn[i*4+1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            
            next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            #next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            
            g2_t = torch.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = self.bn[i*4+3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            
            h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            #h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)
            
            prev_state2 = (1 - g2_t)*prev_state2 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = F.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            prev_state2 = self.n[timestep]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        return prev_state2


class hConvGRU(nn.Module):

    def __init__(self, timesteps=8, filt_size = 9):
        super().__init__()
        self.timesteps = timesteps
        
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, padding=3)
        part1 = np.load("gabor_serre.npy")
        self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = hConvGRUCell(25, 25, filt_size)
        print("Training with filter size:",filt_size,"x",filt_size)
        self.unit1.train()
        
        #self.bn = nn.GroupNorm(25, 25, eps=1e-03)
        self.bn = nn.BatchNorm2d(25, eps=1e-03)
        
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, 0)
        
        self.maxpool = nn.MaxPool2d(150, stride=1)
        
        #self.bn2 = nn.GroupNorm(2, 2, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-03)
        
        self.fc = nn.Linear(2, 2)
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        internal_state = None
        #import pdb; pdb.set_trace()
        #print(x.shape)
        x = self.conv0(x)
        x = torch.pow(x, 2)
        
        for i in range(self.timesteps):
            internal_state  = self.unit1(x, internal_state, timestep=i)
        #import pdb; pdb.set_trace()
        output = self.bn(internal_state)
        output = F.leaky_relu(self.conv6(output))
        output = self.maxpool(output)
        output = self.bn2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
