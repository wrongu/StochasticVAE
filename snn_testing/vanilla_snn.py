import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import mlflow
import os


class StochasticNN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar = -2.5):
        super(StochasticNN, self).__init__()

        """
        Stochastic Neural network for Iris data from data directory with 4 input features
        used for classification for 3 labels. 
        """
        self.user_input_logvar = user_input_logvar
        # 4 -> 20 -> 20 -> 3

        # input layer
        self.weights_mean_ipl = nn.Parameter(torch.Tensor(20, input_dim))
        self.weights_logvar_ipl = nn.Parameter(torch.Tensor(20, input_dim))
        self.bias_mean_ipl = nn.Parameter(torch.Tensor(20))
        self.bias_logvar_l1_ipl = nn.Parameter(torch.Tensor(20))

        # layer 1
        self.weights_mean_l1 = nn.Parameter(torch.Tensor(20, 20))
        self.weights_logvar_l1 = nn.Parameter(torch.Tensor(20, 20))
        self.bias_mean_l1 = nn.Parameter(torch.Tensor(20))
        self.bias_logvar_l1 = nn.Parameter(torch.Tensor(20))

        # output layer
        self.weights_mean_opl = nn.Parameter(torch.Tensor(z_dim, 20))
        self.weights_logvar_opl = nn.Parameter(torch.Tensor(z_dim, 20))
        self.bias_mean_opl = nn.Parameter(torch.Tensor(z_dim))
        self.bias_logvar_opl = nn.Parameter(torch.Tensor(z_dim))

        self.initialize_parameters()        # weights and biases are initialized to a certain value


    def initialize_parameters(self):
        # Initialize means with a normal distribution and standard deviations with a small positive constant
        # Input layer initialization (weights and biases)
        init.kaiming_normal_(self.weights_mean_ipl, mode='fan_in', nonlinearity='relu')
        init.constant_(self.weights_logvar_ipl,self.user_input_logvar)  # small positive std deviation for stochasticity
        init.constant_(self.bias_mean_ipl, 0)      # bias can be initialized to 0 or small value
        init.constant_(self.bias_logvar_l1_ipl, self.user_input_logvar)  # small positive std deviation for stochasticity

        # Hidden layer 1 initialization (weights and biases)
        init.kaiming_normal_(self.weights_mean_l1, mode='fan_in', nonlinearity='relu')
        init.constant_(self.weights_logvar_l1, self.user_input_logvar)  
        init.constant_(self.bias_mean_l1, 0)      
        init.constant_(self.bias_logvar_l1, self.user_input_logvar)     

        # Output layer initialization (weights and biases)
        init.kaiming_normal_(self.weights_mean_opl, mode='fan_in', nonlinearity='relu')
        init.constant_(self.weights_logvar_opl, self.user_input_logvar)  
        init.constant_(self.bias_mean_opl, 0)      
        init.constant_(self.bias_logvar_opl, self.user_input_logvar)     


    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def forward(self, x):
        Softplus = torch.nn.Softplus() 
        # Input Layer
        weights_ipl = self.reparameterization_trick(self.weights_mean_ipl, self.weights_logvar_ipl)
        bias_ipl = self.reparameterization_trick(self.bias_mean_ipl, self.bias_logvar_l1_ipl)
        x = torch.matmul(weights_ipl, x) + bias_ipl.unsqueeze(dim=1)
        x =  Softplus(x)

        # Hidden Layer 1
        weights_l1 = self.reparameterization_trick(self.weights_mean_l1, self.weights_logvar_l1)
        bias_l1 = self.reparameterization_trick(self.bias_mean_l1, self.bias_logvar_l1)
        x = torch.matmul(weights_l1, x) + bias_l1.unsqueeze(dim=1)
        x = Softplus(x)  

        # Output Layer
        weights_opl = self.reparameterization_trick(self.weights_mean_opl, self.weights_logvar_opl)
        bias_opl = self.reparameterization_trick(self.bias_mean_opl, self.bias_logvar_opl)
        x = torch.matmul(weights_opl, x) + bias_opl.unsqueeze(dim=1)
        # x = torch.softmax(x, dim=-1)

        return x
