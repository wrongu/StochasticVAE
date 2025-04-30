from pathlib import Path

import mlflow
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nn_lib.utils import search_runs_by_params
from torchvision import datasets, transforms

from main import (
    ENCODER_PLAN,
    DECODER_PLAN,
    MLFLOW_TRACKING_URI,
    DATA_ROOT,
    MLFLOW_EXPERIMENT,
    get_best_checkpoint_for_run,
)
from probability import log_prob_diagonal_gaussian, logvar_from_fisher
from stochastic_density_network import Stochastic_Density_NN
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_vae import Stochastic_VAE

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
runs = search_runs_by_params(
    experiment_name=MLFLOW_EXPERIMENT,
    params={
        "latent_dim": 5,
        "decoder_source": "ba002b451919474c807c5ed52766eb93",
        "learning_rate": 1e-3,
    },
)

print("Found runs with lambdas:", *runs["params.lambda_"].to_numpy(), sep="\n")

# %%

svaes = []
for idx, run in runs.iterrows():
    svae = Stochastic_VAE(
        Stochastic_Recognition_NN(
            input_dim=784,
            latent_dim=int(run["params.latent_dim"]),
            user_input_logvar=float(run["params.user_input_logvar"]),
            plan=ENCODER_PLAN,
        ),
        Stochastic_Density_NN(
            input_dim=784,
            latent_dim=int(run["params.latent_dim"]),
            plan=DECODER_PLAN,
        ),
        lambda_=float(run["params.lambda_"]),
        lr=float(run["params.learning_rate"]),
        k_neighbor=int(run["params.number_of_nearest_neighbors"]),
        n_forward=int(run["params.n_forward_pass"]),
    )
    checkpoint = torch.load(get_best_checkpoint_for_run(run["run_id"]))
    svae.load_state_dict(checkpoint["state_dict"])
    svaes.append(svae.eval().to("cuda"))

# %%

test_dataset = datasets.MNIST(
    root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=4)

x, y = next(iter(test_loader))
x = x.to("cuda")
print("test classes:", *y)

# %%

mu_zs = []
logvar_zs = []
for svae in svaes:
    with torch.no_grad():
        print(f"Running SVAE with lambda {svae.lambda_}")
        stuff = svae.loss(x)
        print("\tFIM:", stuff["fim_term"].item())
        print(
            "\tAvg Std:",
            logvar_from_fisher(stuff["fim_term"], d=svae.encoder.d).exp().mean().sqrt().item(),
        )
        print("\tEntropy:", stuff["entropy_term"].item())
        mu_z, logvar_z = svae.vmap_encoder(x.unsqueeze(1).expand(-1, svae.n_forward, -1, -1, -1))
        mu_zs.append(mu_z)
        logvar_zs.append(logvar_z)

# %% plotgrid of 2d gaussians for each SVAE


def plot_gauss_mixture_2d(mu, logvar, lim=3, resolution=500, ax=None, **kwargs):
    ax = ax or plt.gca()
    z = torch.linspace(-lim, lim, resolution, device=mu.device)
    dz = z[1] - z[0]
    z1, z2 = torch.meshgrid(z, z)
    z = torch.stack([z1.flatten(), z2.flatten()], dim=1)
    # Get a (10000, num gaussians) grid of log probs
    logvar = torch.clip(logvar, min=torch.log(dz**2 / 4))
    log_prob = log_prob_diagonal_gaussian(z[:, None, :], mu[None, :, :], logvar[None, :, :], dim=2)

    # Sum over the components
    log_prob = torch.logsumexp(log_prob, dim=1)

    # Normalize and reshape back to a grid
    prob = torch.exp(log_prob - torch.logsumexp(log_prob, dim=0)).reshape(resolution, resolution)

    # Contour plot
    ax.contour(z1.cpu(), z2.cpu(), prob.cpu(), levels=[prob.cpu().max() / 2], **kwargs)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# plot_gauss_mixture_2d(mu_zs[0][0, :, :2], logvar_zs[0][0, :, :2], ax=ax, colors="k", alpha=0.1)
# plt.tight_layout()
# plt.show()

for data_idx in range(len(x)):
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    colors = plt.get_cmap("tab10")
    for i in range(4):
        for j in range(4):
            if i < j:
                ax[i, j].remove()
                continue
            for s, (svae, mu_z, logvar_z) in enumerate(zip(svaes, mu_zs, logvar_zs)):
                plot_gauss_mixture_2d(
                    mu_z[data_idx, :, [j, i + 1]],
                    logvar_z[data_idx, :, [j, i + 1]],
                    ax=ax[i, j],
                    colors=colors(s),
                )
            ax[i, j].set_xlabel("$z_{}$".format(j + 1))
            ax[i, j].set_ylabel("$z_{}$".format(i + 2))
    ax[0, 0].legend(
        handles=[
            Line2D([0], [0], color=colors(s), label=f"λ={svae.lambda_}") for s, svae in enumerate(svaes)
        ],
        loc="upper right",
    )
    plt.tight_layout()
    plt.show()
