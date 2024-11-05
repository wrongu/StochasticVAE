import torch
import torch.nn as nn
from training_config import LATENT_DIM


class Stochastic_VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu_z, logvar_z):
        """Sample from the approximate posterior using the reparameterization trick."""

        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)

        z = mu_z + eps * std

        return z

    def loss(self, x):
        mu_z, logvar_z = self.encoder(x)

        z = self.reparameterize(mu_z, logvar_z)
        z = z.view((-1, LATENT_DIM))

        x_recon = self.decoder(z)

        reconstruction_term = self.decoder.log_likelihood(x, x_recon).sum()
        kl_term = self.encoder.kl(mu_z, logvar_z)  # shape is []
        # elbo = reconstruction_term - kl_term
        loss = kl_term - reconstruction_term

        return loss.sum(), kl_term, reconstruction_term, x_recon
