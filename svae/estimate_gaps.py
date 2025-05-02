from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from main import ENCODER_PLAN, DECODER_PLAN, get_best_checkpoint_for_run, DATA_ROOT
from probability import (
    log_prob_diagonal_gaussian,
    reparameterization_trick,
    kl_q_prior,
)
from stochastic_density_network import Stochastic_Density_NN, PixelCovariance
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_vae import Stochastic_VAE
from tqdm.auto import tqdm


def stats(values: torch.Tensor):
    mean = torch.mean(values)
    standard_deviation = torch.std(values)
    standard_error = standard_deviation / np.sqrt(len(values))
    return mean, standard_deviation, standard_error


def estimate_elbos(
    model: Stochastic_VAE,
    data: DataLoader,
    n_z: int = 1,
    n_iter_per_item: int = 100,
    k_iwae: int = 10,
    device="cuda",
):
    """Estimate the amortization gap for a given VAE model and dataset."""
    assert model.encoder.deterministic, "Encoder must be deterministic (VAEs only)"

    model.eval().to(device)

    model_elbos = []
    optimized_elbos = []
    iwae_elbos = []
    for batch in tqdm(data, desc="Batches", position=0):
        x = batch[0].to(device)
        with torch.no_grad():
            mu_z, logvar_z = model.encoder(x)

        def elbo_helper(x_, mu, logvar, n_samples: int = 1):
            kl_q_p = kl_q_prior(mu, logvar)
            z = reparameterization_trick(
                mu, logvar, n_samples=n_samples, stack_dim=model.n_forward_dim
            )
            x_recon = model.vmap_decoder(z)
            log_p_x_given_z = model.decoder.log_likelihood(
                torch.stack([x_] * n_samples, dim=model.n_forward_dim), x_recon, flatten_dim=2
            )
            return log_p_x_given_z.mean(dim=model.n_forward_dim) - kl_q_p

        model_elbos.append(elbo_helper(x, mu_z, logvar_z, n_z))

        mu_z.requires_grad = True
        logvar_z.requires_grad = True
        optim = torch.optim.SGD([mu_z, logvar_z], lr=1e-3, momentum=0.5)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
        history = defaultdict(list)
        for itr in range(n_iter_per_item):
            elbo = elbo_helper(x, mu_z, logvar_z, n_z)
            optim.zero_grad()
            (-elbo.sum()).backward()
            # Manage gradient scale with natural gradient; changes in mu are "bigger" when logvar
            # is small. Preconditioner is the precision matrix of q.
            with torch.no_grad():
                precision = torch.exp(-logvar_z)
                mu_z.grad /= precision
            optim.step()
            sched.step()

        #     history["elbo"].append(elbo.detach().clone())
        #     history["mu_z"].append(mu_z[:, 0].detach().clone())
        #     history["logvar_z"].append(logvar_z[:, 0].detach().clone())
        #
        # # DEBUG PLOTS
        # plt.plot(torch.stack(history["elbo"]).cpu())
        # plt.title("ELBO over iterations")
        # plt.show()
        #
        # plt.plot(torch.stack(history["mu_z"]).cpu())
        # plt.title("mu_z[0] over iterations")
        # plt.show()
        #
        # plt.plot(torch.stack(history["logvar_z"]).cpu())
        # plt.title("logvar_z[0] over iterations")
        # plt.show()
        #
        # exit(0)

        # Recompute the ELBO with the updated parameters
        mu_z.requires_grad = False
        logvar_z.requires_grad = False
        optimized_elbos.append(elbo_helper(x, mu_z, logvar_z, n_z))

        # Compute the IWAE ELBO using a slightly inflated variance
        iwae_elbos.append(elbo_iwae(model, x, mu_z, logvar_z + 0.1, k=k_iwae))

    return (
        torch.cat(model_elbos).detach().cpu(),
        torch.cat(optimized_elbos).detach().cpu(),
        torch.cat(iwae_elbos).detach().cpu(),
    )


def elbo_iwae(model: Stochastic_VAE, x_, mu, logvar, k: int = 1):
    batch_size = x_.size(0)

    # Sample k times from q
    z = reparameterization_trick(mu, logvar, n_samples=k, stack_dim=model.n_forward_dim)

    # Run the stack of zs through the vmapped decoder
    x_recon = model.vmap_decoder(z)

    # Compare reconstructions with stack of inputs
    x_stack = torch.stack([x_] * k, dim=model.n_forward_dim)
    log_p_x_given_z = model.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2)
    assert log_p_x_given_z.shape == (batch_size, k)

    # Calculate log prior on each z
    log_p_z = log_prob_diagonal_gaussian(z, torch.zeros_like(z), torch.zeros_like(z), dim=-1)
    assert log_p_z.shape == (batch_size, k)

    # Calculate log q on each z
    log_q_z = log_prob_diagonal_gaussian(
        z, mu.unsqueeze(model.n_forward_dim), logvar.unsqueeze(model.n_forward_dim), dim=-1
    )
    assert log_q_z.shape == (batch_size, k)

    # Calculate ELBO using logsumexp trick
    elbo_iwae = torch.logsumexp(log_p_z + log_p_x_given_z - log_q_z, dim=1) - torch.log(
        torch.as_tensor(k, device=x_.device)
    )
    assert elbo_iwae.shape == (batch_size,)

    return elbo_iwae


if __name__ == "__main__":
    mlflow.set_tracking_uri("/data/projects/SVAE/mlruns")
    the_run = mlflow.get_run("aee20ec5d660489eae8f5c85832362ed")
    k_iwae = 100

    encoder = Stochastic_Recognition_NN(
        input_dim=784, latent_dim=5, user_input_logvar=-np.inf, plan=ENCODER_PLAN
    )
    decoder = Stochastic_Density_NN(
        input_dim=784, latent_dim=5, plan=DECODER_PLAN, pixel_covariance=PixelCovariance.ISOTROPIC
    )
    svae = Stochastic_VAE(encoder, decoder, lambda_=np.inf, lr=0.0, k_neighbor=1, n_forward=1)
    checkpoint = torch.load(get_best_checkpoint_for_run(the_run.info.run_id))
    svae.load_state_dict(checkpoint["state_dict"])

    test_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=4
    )

    model_elbos, optimized_elbos, iwae_elbos = estimate_elbos(
        svae, test_loader, n_z=5, n_iter_per_item=100, k_iwae=k_iwae, device="cuda"
    )
    amortization_gap = optimized_elbos - model_elbos
    approximation_gap = iwae_elbos - optimized_elbos
    total_gap = iwae_elbos - model_elbos

    bin_edges = np.linspace(
        *torch.quantile(
            torch.concat([amortization_gap, approximation_gap, total_gap]),
            torch.as_tensor([0.05, 0.95]),
        ),
        51,
    )

    plt.figure()
    plt.hist(amortization_gap, bins=bin_edges, density=True, label="Amortization Gap", alpha=0.5)
    plt.hist(approximation_gap, bins=bin_edges, density=True, label="Approximation Gap", alpha=0.5)
    plt.hist(total_gap, bins=bin_edges, density=True, label="Total Gap", alpha=0.5)
    plt.legend()
    plt.xlabel("∆ELBO")
    plt.show()

    model_avg, model_std, _ = stats(model_elbos)
    optimized_avg, optimized_std, _ = stats(optimized_elbos)
    iwae_avg, iwae_std, _ = stats(iwae_elbos)

    plt.figure()
    plt.plot([0, 1, 2], np.stack([model_elbos, optimized_elbos, iwae_elbos]), alpha=0.1, color="k")
    plt.boxplot([model_elbos, optimized_elbos, iwae_elbos], positions=[0, 1, 2], showfliers=False)
    plt.xticks([0, 1, 2], ["Amortized q", "Optimized q", "'True' posterior"])
    plt.ylabel("ELBO")
    plt.show()
