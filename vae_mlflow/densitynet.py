import torch
from torch import nn
import torch.nn.functional as F
from training_config import H1_DIM, H2_DIM, H3_DIM, H4_DIM, LATENT_DIM

class DensityNet(nn.Module):


    def __init__(self, latent_dim: int = 10, input_dim: int = 784):
        super(DensityNet, self).__init__()

        # Boilerplate NN stuff
        self.fc5 = nn.Linear(LATENT_DIM, H4_DIM)
        self.fc6 = nn.Linear(H4_DIM, H3_DIM)
        self.fc7 = nn.Linear(H3_DIM, H2_DIM)
        self.fc8 = nn.Linear(H2_DIM, H1_DIM)
        self.fc9 = nn.Linear(H1_DIM, input_dim)

        # Add a diagonal covariance in pixel space
        self.logvar_x = nn.Parameter(torch.zeros(input_dim))

    def log_likelihood_gaussian(self, x, mu_z, logvar_z):
      return -0.5 * (logvar_z + (x - mu_z)**2 / logvar_z.exp()).sum(dim=-1)


    def log_likelihood(self, x, recon_x):
        """Calculate p( x|mu,Sigma) for a gaussian with diagonal covariance."""
        return self.log_likelihood_gaussian(x, recon_x, self.logvar_x)


    def forward(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        # h9 = F.relu(self.fc9(h8))

        recon_x = torch.sigmoid(self.fc9(h8))

        return recon_x