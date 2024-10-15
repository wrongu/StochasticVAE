import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal, MixtureSameFamily, Categorical


LOGPI = np.log(np.pi)


def entropy_monte_carlo(p: Distribution, n: int):
    x = p.sample((n,))
    logp = -p.log_prob(x)
    return logp.mean(), logp.std() / n**0.5


def kth_nearest_neighbor_dist(x, k=1):
    """Given x, a (n x d) tensor of n points in d dimensions, calculate the nxn pairwise distances
    between rows of x, then get the kth smallest distance for each point in x.

    Returns a (n,) tensor of the kth nearest neighbor distance for each point in x.
    """
    # xxT = x @ x.t()
    xxT = torch.einsum("i...,j...->ij", x, x)
    sq_pair_dist = torch.diagonal(xxT, 0)[:, None] + torch.diagonal(xxT, 0)[None, :] - 2 * xxT
    return torch.kthvalue(sq_pair_dist, k+1, dim=1).values ** 0.5


def entropy_singh_2003(p: Distribution, n: int, k: int):
    x = p.sample((n,))
    d = torch.tensor(p.event_shape[0])  # dimensionality of the distribution
    knn_dist = kth_nearest_neighbor_dist(x, k)
    n = torch.tensor(n)
    k = torch.tensor(k)
    log_numerator = torch.log(k) + torch.lgamma(d / 2 + 1)
    log_denominator = torch.log(n) + d / 2 * LOGPI + d * torch.log(knn_dist)
    bias_correction = torch.log(k) - torch.digamma(k)
    terms = log_numerator - log_denominator
    return -terms.mean() + bias_correction, terms.std() / n**0.5


def do_entropy_compare(p: Distribution, ns, ks):
    h_mc = np.zeros((len(ns), 2))
    h_singh = [np.zeros((len(ns), 2)) for _ in ks]
    for i, n in enumerate(ns):
        h_mc[i] = entropy_monte_carlo(p, n)
        for j, k in enumerate(ks):
            if k+1 > n:
                h_singh[j][i] = np.nan, np.nan
            else:
                h_singh[j][i] = entropy_singh_2003(p, n, k)
                print("n: " + str(n)+ " k: " + str(k) + str(h_singh[j][i]))
                print()

    return h_mc, h_singh


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    
    # multivariate distribution with mean 0 [2x2], covariance identity matrix [2x2] 
    p1 = MultivariateNormal(torch.zeros(2), torch.eye(2))  
    p2 = MixtureSameFamily(
        Categorical(torch.tensor([0.5, 0.5, 0.5, 0.5])),
        MultivariateNormal(torch.randn(4, 2), torch.stack([torch.eye(2)]*4, dim=0)),
    )

    ns = np.logspace(0, 4, 5).astype(int)
    ks = np.arange(1, 4)
    h_mc_1, h_singh_1 = do_entropy_compare(p1, ns, ks)
    h_mc_2, h_singh_2 = do_entropy_compare(p2, ns, ks)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, (h_mc, h_singh) in enumerate([(h_mc_1, h_singh_1), (h_mc_2, h_singh_2)]):
        ax[i].plot(ns, h_mc[:, 0], label="Monte Carlo")
        ax[i].fill_between(ns, h_mc[:, 0] - h_mc[:, 1], h_mc[:, 0] + h_mc[:, 1], alpha=0.3)
        for j, k in enumerate(ks):
            ax[i].plot(ns, h_singh[j][:, 0], label=f"Singh (k={k})")
            ax[i].fill_between(
                ns,
                h_singh[j][:, 0] - h_singh[j][:, 1],
                h_singh[j][:, 0] + h_singh[j][:, 1],
                alpha=0.3,
            )
        ax[i].set_xscale("log")
        ax[i].set_xlabel("n")
        ax[i].set_ylabel("Entropy")
        ax[i].legend()
    fig.tight_layout()

    plt.savefig('entropy_estimator/Entropy Plot')
