import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from training_config import PLAN

class Stochastic_Density_NN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar = -2.5):
        super(Stochastic_Density_NN, self).__init__()
        
        self.user_input_logvar = user_input_logvar

        for plan_idx in range(len(PLAN)):
            if plan_idx == 0:
                self.weights_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN[plan_idx], input_dim)))
                self.weights_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN[plan_idx], input_dim)))
                
                self.bias_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN[plan_idx])))
                self.bias_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN[plan_idx])))

            elif plan_idx == len(PLAN) - 1:
                self.weights_mean.append(nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(z_dim)))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(z_dim)))
            
            else:
                self.weights_mean.append(nn.Parameter(torch.Tensor(PLAN[plan_idx], PLAN[plan_idx - 1])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(PLAN[plan_idx], PLAN[plan_idx - 1])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(PLAN[plan_idx])))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(PLAN[plan_idx])))

        self.initialize_parameters()


    def initialize_parameters(self):

        for i, layer in enumerate(self.weights_mean):
            init.kaiming_normal_(layer, mode='fan_in', nonlinearity='relu')

        for i, layer in enumerate(self.weights_logvar):
            init.constant_(layer,self.user_input_logvar) 

        for i, layer in enumerate(self.bias_mean):
            init.constant_(layer, 0)

        for i, layer in enumerate(self.bias_logvar):
            init.constant_(layer, self.user_input_logvar)


    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return z
    
                