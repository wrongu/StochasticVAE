import torch
import torch.nn as nn
import torch.nn.init as init
from training_config import PLAN_DECODER

class Stochastic_Density_NN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar = -2.5):
        super(Stochastic_Density_NN, self).__init__()
        
        self.user_input_logvar = user_input_logvar

        for plan_idx in range(len(PLAN_DECODER)):
            #input layer
            if plan_idx == 0:
                self.weights_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx], z_dim)))
                self.weights_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx], z_dim)))
                
                self.bias_mean = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx])))
                self.bias_logvar = nn.ModuleList().append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx])))

            #output layer 
            elif plan_idx == len(PLAN_DECODER) - 1:
                self.weights_mean.append(nn.Parameter(torch.Tensor(input_dim, PLAN_DECODER[plan_idx])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(input_dim, PLAN_DECODER[plan_idx])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(input_dim)))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(input_dim)))
            
            else:
                self.weights_mean.append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx], PLAN_DECODER[plan_idx - 1])))
                self.weights_logvar.append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx], PLAN_DECODER[plan_idx - 1])))

                self.bias_mean.append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx])))
                self.bias_logvar.append(nn.Parameter(torch.Tensor(PLAN_DECODER[plan_idx])))

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
    
                