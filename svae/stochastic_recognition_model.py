import torch
import torch.nn as nn
import torch.nn.functional as F
from training_config import PLAN, DEVICE
from contextlib import contextmanager, ExitStack


class StochasticLinear(nn.Linear):
    def __init__(self, init_logvar: float = -2.5, *args, **kwargs):
        super(StochasticLinear, self).__init__(*args, **kwargs)

        # Create _mean and _logvar versions of each parameter
        self.weights_mean = nn.Parameter(self.weight.data.clone())
        self.weights_logvar = nn.Parameter(torch.ones_like(self.weight) * init_logvar)
        self._has_bias = self.bias is not None
        if self._has_bias:
            self.bias_mean = nn.Parameter(self.bias.data.clone())
            self.bias_logvar = nn.Parameter(torch.ones_like(self.bias) * init_logvar)

        # Remove old parameters
        delattr(self, 'weight')
        delattr(self, 'bias')

        # Keep track of self._eps for reparameterization trick and freezing
        self._weight_eps = None
        self._bias_eps = None
        self.resample()

        self._frozen = False

    @contextmanager
    def frozen(self):
        self._frozen = True
        yield
        self._frozen = False

    def resample(self):
        self._weight_eps = torch.randn_like(self.weights_mean)
        self._bias_eps = torch.randn_like(self.bias_mean) if self._has_bias else None

    def forward(self, x):
        if not self._frozen:
            self.resample()

        weight = self.weights_mean + self.weights_logvar.exp() * self._weight_eps
        bias = self.bias_mean + self.bias_logvar.exp() * self._bias_eps if self._has_bias else None
        return F.linear(x, weight, bias)


class Stochastic_Recognition_NN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar=-2.5):
        super(Stochastic_Recognition_NN, self).__init__()

        self.norms = nn.ModuleList()
        layers = [StochasticLinear(user_input_logvar, input_dim, PLAN[0]), nn.LayerNorm(PLAN[0]), nn.ReLU()]
        for in_size, out_size in zip(PLAN[:-1], PLAN[1:]):
            layers.append(StochasticLinear(user_input_logvar, in_size, out_size))
            layers.append(nn.LayerNorm(out_size))
            layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*layers)
        self.head_mu_z = StochasticLinear(user_input_logvar, PLAN[-1], z_dim)
        self.head_logvar_z = StochasticLinear(user_input_logvar, PLAN[-1], z_dim)
        self._eps = None
        self.z_dim = z_dim
        self.resample()
        self._frozen = False

    def resample(self):
        self._eps = torch.randn(1, self.z_dim)

    @contextmanager
    def frozen_weights(self):
        with ExitStack() as stack:
            for _, module in self.named_modules():
                if isinstance(module, StochasticLinear):
                    stack.enter_context(module.frozen())
            yield

    @contextmanager
    def frozen_z(self):
        self._frozen = True
        yield
        self._frozen = False


    def reparameterization_trick(self, mu, logvar):
        if not self._frozen:
            self.resample()
        std = torch.exp(0.5 * logvar)
        return mu + self._eps * std

    def kl(self, mu_z, logvar_z):
        """Calculate KL divergence between a diagonal gaussian and a standard normal."""
        return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        mean_z = self.head_mu_z(x)
        logvar_z = self.head_logvar_z(x)
        return mean_z, logvar_z


if __name__ == "__main__":
    my_model = Stochastic_Recognition_NN(input_dim=28*28, z_dim=10)

    x = torch.randn(10, 1, 28, 28)
    with my_model.frozen_weights(), my_model.frozen_z():
        out1 = my_model(x)
        z1 = my_model.reparameterization_trick(*out1)
        out2 = my_model(x)
        z2 = my_model.reparameterization_trick(*out2)

    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out1[1], out2[1])
    assert torch.allclose(z1, z2)

    with my_model.frozen_weights():
        out3 = my_model(x)
        z3 = my_model.reparameterization_trick(*out3)
        out4 = my_model(x)
        z4 = my_model.reparameterization_trick(*out4)

    assert not torch.allclose(z3, z4)
    assert torch.allclose(out3[0], out4[0])
    assert torch.allclose(out3[1], out4[1])

    with my_model.frozen_z():
        out5 = my_model(x)
        z5 = my_model.reparameterization_trick(*out5)
        out6 = my_model(x)
        z6 = my_model.reparameterization_trick(*out6)
        z7 = my_model.reparameterization_trick(*out6)

    assert not torch.allclose(z5, z6)
    assert torch.allclose(z6, z7)
    assert not torch.allclose(out5[0], out6[0])
    assert not torch.allclose(out5[1], out6[1])
