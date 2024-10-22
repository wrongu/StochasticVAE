import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt


class Stochastic_VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def loss(self):
        pass