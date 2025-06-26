import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class AnomalyLayer(nn.Module):
    def __init__(self, dim):
        super(AnomalyLayer, self).__init__()
        self.vars = nn.ParameterList()
        self.config = [
            ('linear', [dim, 2 * dim]),  # 输入维度变为 2 * dim
            ('linear', [1, dim])
        ]
        for i, (name, param) in enumerate(self.config):
            w = nn.Parameter(torch.ones(*param))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(param[0])))

    def forward(self, x1, x2, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        combined_x = torch.cat((x1, x2), dim=1)  # 将 x1 和 x2 连接起来
        idx = 0
        for name, param in self.config:
            w, b = vars[idx], vars[idx + 1]
            combined_x = F.linear(combined_x, w, b)
            idx += 2

        assert idx == len(vars)

        return combined_x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class graph_deviation_network(torch.nn.Module):
    def __init__(self, args, device):
        super(graph_deviation_network, self).__init__()
        self.net = AnomalyLayer(args.hidden_dim)

        self.memory_size = args.memory_size 
        self.sample_size = args.sample_size 
        self.device = device
        self.memory = torch.randn_like(torch.zeros(self.memory_size, dtype=torch.float32, requires_grad=False)).to(device)
        self.time_memory = torch.zeros(self.memory_size, dtype=torch.float32, requires_grad=False).to(device)
        self.idx = 0
        self.fc1 = torch.nn.Linear(1, 1)

    def dev_loss(self, y_true, y_prediction, current_time):


        index = torch.LongTensor(random.sample(range(self.memory_size), self.sample_size)).to(self.device)
        ref = torch.index_select(self.memory, 0, index)
        ref_time = torch.index_select(self.time_memory, 0, index)

        lambda_t = 1e-7
        time_diff = (torch.abs(current_time.reshape([-1, 1]) - ref_time.reshape([1,-1])) )
        w_time = 1 / (torch.log(lambda_t * time_diff +1) +1)
        
        mean_value = torch.sum(torch.mul(w_time, ref), dim=1) / torch.sum(w_time, dim=1)
        std_value =  torch.pow((ref.reshape([1,-1]) - mean_value.reshape([-1, 1])), 2)
        std_value = torch.sum(torch.mul(w_time, std_value), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.sqrt(std_value)
        dev = (y_prediction - mean_value) / (std_value +1e-6)
        inlier_loss = torch.abs(dev)
        outlier_loss = 5 - torch.abs(dev)
        outlier_loss[outlier_loss < 0.] = 0
        loss = (1 - y_true) * inlier_loss.flatten() + y_true * outlier_loss.flatten()
        #print(loss)
        return loss

    def dev_diff(self, y_prediction, current_time, label):
        index = torch.LongTensor(random.sample(range(self.memory_size), self.sample_size)).to(self.device)
        ref = torch.index_select(self.memory, 0, index)
        ref_time = torch.index_select(self.time_memory, 0, index)
        lambda_t = 1e-7
        time_diff = (torch.abs(current_time.reshape([-1, 1]) - ref_time.reshape([1, -1])))
        w_time = 1 / (torch.log(lambda_t * time_diff + 1) + 1)

        mean_value = torch.sum(torch.mul(w_time, ref), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.pow((ref.reshape([1, -1]) - mean_value.reshape([-1, 1])), 2)
        std_value = torch.sum(torch.mul(w_time, std_value), dim=1) / torch.sum(w_time, dim=1)
        std_value = torch.sqrt(std_value)

        dev = (y_prediction - torch.mean(ref)) / torch.std(ref)



        group = torch.where(dev > -3, torch.ones_like(y_prediction), torch.zeros_like(y_prediction))
        group = torch.where(dev > -2, group+1, group)
        group = torch.where(dev > -1, group + 1, group)
        group = torch.where(dev > 1, group + 1, group)
        group = torch.where(dev > 2, group + 1, group)
        group = torch.where(dev > 3, group + 1, group)

        return dev, group


    def forward(self, x1, x2, time, label=None):
        ana_score = self.net(x1, x2)

        if self.training:
            record_mem = (ana_score.clone().detach())[label <= 0]
            record_time_mem = (time.clone().detach())[label <= 0]
            self.idx_after = self.idx + record_mem.shape[0]
            if self.idx_after < self.memory_size:
                self.memory[self.idx:self.idx_after] = record_mem.reshape([-1])
                self.time_memory[self.idx:self.idx_after] = record_time_mem.reshape([-1])
            elif self.idx_after >= self.memory_size:
                self.memory[self.idx:self.memory_size] = record_mem.reshape([-1])[0:(self.memory_size - self.idx)]
                self.memory[0:self.idx_after % self.memory_size] = record_mem.reshape([-1])[
                                                                   (self.memory_size - self.idx):]
                self.time_memory[self.idx:self.memory_size] = record_time_mem.reshape([-1])[0:(self.memory_size - self.idx)]
                self.time_memory[0:self.idx_after % self.memory_size] = record_time_mem.reshape([-1])[
                                                                   (self.memory_size - self.idx):]

            self.idx = self.idx_after % self.memory_size

        return ana_score
