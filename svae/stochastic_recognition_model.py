import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt


class Stochastic_Recognition_NN(nn.Module):

    PLAN = [500, 300, 200, 100, 50]

    def __init__(self, input_dim, z_dim, user_input_logvar = -2.5):
        super(Stochastic_Recognition_NN, self).__init__()
        
        self.user_input_logvar = user_input_logvar

        for plan_idx in range(len(self.PLAN)):
            if plan_idx == 0:
                self.weights_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx], input_dim)))
                self.weights_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx], input_dim)))
                
                self.bias_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx])))
                self.bias_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx])))

            elif plan_idx == len(self.PLAN) - 1:
                self.weights_mean.append(nn.Parameter(torch.Tensor(z_dim, self.PLAN[plan_idx])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(z_dim, self.PLAN[plan_idx])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(z_dim)))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(z_dim)))
                
            
            else:
                self.weights_mean.append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx], self.PLAN[plan_idx - 1])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx], self.PLAN[plan_idx - 1])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx])))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(self.PLAN[plan_idx])))

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
    
    def forward(self, x):
        Softplus = torch.nn.Softplus()

        for i in range(len(self.weights_mean)):
            weights_ipl = self.reparameterization_trick(self.weights_mean[i], self.weights_logvar[i])
            bias_ipl = self.reparameterization_trick(self.bias_mean[i], self.bias_logvar[i])
            x = torch.matmul(weights_ipl, x) + bias_ipl.unsqueeze(dim=1)  
            if i == len(self.weights_mean) - 1:
                x =  Softplus(x)

        return x                
           
           











        
