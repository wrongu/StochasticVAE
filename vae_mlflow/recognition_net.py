import torch
from torch import nn
import torch.nn.functional as F
from training_config import H1_DIM, H2_DIM, H3_DIM, H4_DIM


class RecognitionModel(nn.Module):

    def __init__(self, latent_dim: int = 10):
        super(RecognitionModel, self).__init__()

        # Boilerplate NN stuff
        self.fc1 = nn.Linear(784, H1_DIM)
        self.fc2 = nn.Linear(H1_DIM, H2_DIM)
        self.fc3 = nn.Linear(H2_DIM, H3_DIM)
        self.fc4 = nn.Linear(H3_DIM, H4_DIM)

        self.fc41 = nn.Linear(H4_DIM, latent_dim)
        self.fc42 = nn.Linear(H4_DIM, latent_dim)

    def kl(self, mu_z, logvar_z):
        """Calculate KL divergence between a diagonal gaussian and a standard normal."""
        return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    def forward(self, x):
        h1 = self.fc1(x)
        h1 = F.relu(h1)

        h2 = self.fc2(h1)
        h2 = F.relu(h2)

        h3 = self.fc3(h2)
        h3 = F.relu(h3)

        h4 = self.fc4(h3)
        h4 = F.relu(h4)

        mu = self.fc41(h4)

        logvar = self.fc42(h4)

        return mu, logvar
