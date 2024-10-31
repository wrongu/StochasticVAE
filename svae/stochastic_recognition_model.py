import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from training_config import PLAN, DEVICE

class Stochastic_Recognition_NN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar = -20):
        super(Stochastic_Recognition_NN, self).__init__()
        
        self.user_input_logvar = user_input_logvar

        self.weights_mean = []
        self.weights_logvar = []
        self.bias_mean = []
        self.bias_logvar = []
        self.norms = nn.ModuleList()

        for plan_idx in range(len(PLAN)+1):

            # Input layer
            if plan_idx == 0:
                w_mean = nn.Parameter(torch.Tensor(PLAN[plan_idx], input_dim))
                w_logvar = nn.Parameter(torch.Tensor(PLAN[plan_idx], input_dim))
                b_mean = nn.Parameter(torch.Tensor(PLAN[plan_idx]))
                b_logvar = nn.Parameter(torch.Tensor(PLAN[plan_idx]))

            # Output layer
            elif plan_idx == len(PLAN):
                #head 1 - mean
                head_1_w_mean = nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx-1]))
                head_1_w_logvar = nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx-1]))
                head_1_b_mean = nn.Parameter(torch.Tensor(z_dim))
                head_1_b_logvar = nn.Parameter(torch.Tensor(z_dim))

                #head 2 - logvar
                head_2_w_mean = nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx-1]))
                head_2_w_logvar = nn.Parameter(torch.Tensor(z_dim, PLAN[plan_idx-1]))
                head_2_b_mean = nn.Parameter(torch.Tensor(z_dim))
                head_2_b_logvar = nn.Parameter(torch.Tensor(z_dim))

                #head 1 - registering the parameters
                self.register_parameter(f"weights_mean_{plan_idx}", head_1_w_mean)
                self.register_parameter(f"weights_logvar_{plan_idx}", head_1_w_logvar)
                self.register_parameter(f"bias_mean_{plan_idx}", head_1_b_mean)
                self.register_parameter(f"bias_logvar_{plan_idx}", head_1_b_logvar)

                #head 2 - registering the parameters
                self.register_parameter(f"weights_mean_{plan_idx}", head_2_w_mean)
                self.register_parameter(f"weights_logvar_{plan_idx}", head_2_w_logvar)
                self.register_parameter(f"bias_mean_{plan_idx}", head_2_b_mean)
                self.register_parameter(f"bias_logvar_{plan_idx}", head_2_b_logvar)

                self.weights_mean.append(head_1_w_mean)
                self.weights_logvar.append(head_1_w_logvar)
                self.bias_mean.append(head_1_b_mean)
                self.bias_logvar.append(head_1_b_logvar)

                self.weights_mean.append(head_2_w_mean)
                self.weights_logvar.append(head_2_w_logvar)
                self.bias_mean.append(head_2_b_mean)
                self.bias_logvar.append(head_2_b_logvar)


                break

            # Hidden layers
            else:
                w_mean = nn.Parameter(torch.Tensor(PLAN[plan_idx], PLAN[plan_idx - 1]))
                w_logvar = nn.Parameter(torch.Tensor(PLAN[plan_idx], PLAN[plan_idx - 1]))
                b_mean = nn.Parameter(torch.Tensor(PLAN[plan_idx]))
                b_logvar = nn.Parameter(torch.Tensor(PLAN[plan_idx]))
            
            self.norms.append(nn.LayerNorm(PLAN[plan_idx]))

            # Register parameters
            self.register_parameter(f"weights_mean_{plan_idx}", w_mean)
            self.register_parameter(f"weights_logvar_{plan_idx}", w_logvar)
            self.register_parameter(f"bias_mean_{plan_idx}", b_mean)
            self.register_parameter(f"bias_logvar_{plan_idx}", b_logvar)

            # Append to lists
            self.weights_mean.append(w_mean)
            self.weights_logvar.append(w_logvar)
            self.bias_mean.append(b_mean)
            self.bias_logvar.append(b_logvar)

        self.initialize_parameters()


    def initialize_parameters(self):
        for i, layer in enumerate(self.weights_mean):
            init.kaiming_normal_(layer, mode='fan_in', nonlinearity='relu')
        for i, layer in enumerate(self.weights_logvar):
            init.constant_(layer, self.user_input_logvar)
        for i, layer in enumerate(self.bias_mean):
            init.constant_(layer, 0)
        for i, layer in enumerate(self.bias_logvar):
            init.constant_(layer, self.user_input_logvar)


    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def kl(self, mu_z, logvar_z):
        """Calculate KL divergence between a diagonal gaussian and a standard normal."""
        return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    

    def forward(self, x):
        for i in range(len(self.weights_mean)-2):
            weights_ipl = self.reparameterization_trick(self.weights_mean[i], self.weights_logvar[i])
            bias_ipl = self.reparameterization_trick(self.bias_mean[i], self.bias_logvar[i])
            x = torch.matmul(x, weights_ipl.T) + bias_ipl.view(1, -1)
            x = self.norms[i](x)
            x = F.relu(x)

        # head 1 - mean prediction
        head_1_weights_ipl = self.reparameterization_trick(self.weights_mean[-2], self.weights_logvar[-2])
        head_1_bias_ipl = self.reparameterization_trick(self.bias_mean[-2], self.bias_logvar[-2])
        head_1_weights_ipl, head_1_bias_ipl = head_1_weights_ipl.to(DEVICE), head_1_bias_ipl.to(DEVICE)
        mean_z = torch.matmul(x, head_1_weights_ipl.T) + head_1_bias_ipl.view(1, -1)

        #head 2- logvar prediction
        head_2_weights_ipl = self.reparameterization_trick(self.weights_mean[-1], self.weights_logvar[-1])
        head_2_bias_ipl = self.reparameterization_trick(self.bias_mean[-1], self.bias_logvar[-1])
        head_2_weights_ipl, head_2_bias_ipl = head_2_weights_ipl.to(DEVICE), head_2_bias_ipl.to(DEVICE)
        logvar_z = torch.matmul(x, head_2_weights_ipl.T) + head_2_bias_ipl.view(1, -1)


        return mean_z, logvar_z
