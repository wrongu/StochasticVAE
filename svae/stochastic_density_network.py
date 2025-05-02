import torch
import torch.nn as nn

from probability import log_prob_diagonal_gaussian
import torch.nn.functional as F
from enum import Enum, auto


class PixelCovariance(Enum):
    IDENTITY = "identity"
    ISOTROPIC = "isotropic"
    DIAGONAL = "diagonal"

    def __str__(self):
        return self.value


class Stochastic_Density_NN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        plan: list[int],
        pixel_covariance: PixelCovariance = PixelCovariance.DIAGONAL,
    ):
        super(Stochastic_Density_NN, self).__init__()
        plan_with_inputs_and_outputs = [latent_dim] + plan + [input_dim]

        self.d = latent_dim
        self.layers = nn.ModuleList()
        for i in range(1, len(plan_with_inputs_and_outputs)):
            in_size = plan_with_inputs_and_outputs[i - 1]
            out_size = plan_with_inputs_and_outputs[i]
            self.layers.append(nn.Linear(in_size, out_size))

        if pixel_covariance == PixelCovariance.IDENTITY:
            self.register_buffer("logvar_x", torch.zeros(input_dim))
        elif pixel_covariance == PixelCovariance.ISOTROPIC:
            self.register_parameter("logvar_x", nn.Parameter(torch.zeros(1)))
        elif pixel_covariance == PixelCovariance.DIAGONAL:
            self.register_parameter("logvar_x", nn.Parameter(torch.zeros(input_dim)))
        else:
            raise ValueError(f"Invalid pixel covariance type: {pixel_covariance}.")

    def log_likelihood(self, x, recon_x, flatten_dim: int = 1):
        """Calculate p( x|mu,Sigma) for a gaussian with diagonal covariance."""
        # Flatten everything
        x = torch.flatten(x, start_dim=flatten_dim)
        recon_x = torch.flatten(recon_x, start_dim=flatten_dim)
        return log_prob_diagonal_gaussian(x, recon_x, self.logvar_x)

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        return torch.sigmoid(self.layers[-1](z))

    @torch.no_grad()
    def generate(self, n: int, pixel_noise: bool = True):
        z = torch.randn(n, self.d, device=self.layers[0].weight.device)
        recon_x = self.forward(z)
        if pixel_noise:
            return recon_x + torch.randn_like(recon_x) * torch.exp(self.logvar_x)
        else:
            return recon_x
