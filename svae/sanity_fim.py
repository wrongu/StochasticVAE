import numpy as np
from torch.distributions import MultivariateNormal
from probability import log_prob_diagonal_gaussian, log_det_fisher
import matplotlib.pyplot as plt
import torch
from tqdm.auto import trange

n = 10000
d = 20

theta = torch.randn(2*d, requires_grad=True)
mu, logvar = theta[:d], theta[d:]
mvn = MultivariateNormal(mu, covariance_matrix=torch.diag(logvar.exp()))
x = mvn.sample((n,)).detach()


# %% log(det(FIM)) test

fim = torch.zeros(2*d, 2*d)
log_probs = mvn.log_prob(x)
for i in trange(n):
    score = torch.autograd.grad(log_probs[i], theta, retain_graph=True)[0]
    fim += torch.outer(score, score) / n

log_det_fim_torch = torch.linalg.slogdet(fim)[1]
log_det_fim_mine = log_det_fisher(mu, logvar)
print("Log det FIM (approximate, torch):", log_det_fim_torch)
print("Log det FIM (analytical, ours):", log_det_fim_mine)

# %% log density test
with torch.no_grad():
    log_prob_torch = mvn.log_prob(x)
    log_prob_mine = log_prob_diagonal_gaussian(x, mu, logvar)

plt.scatter(log_prob_torch, log_prob_mine - d / 2 * np.log(2 * np.pi), marker=".")
plt.axis("equal")
plt.grid()
plt.xlabel("log prob MVN (torch)")
plt.ylabel("log prob MVN (mine)")
plt.title("Log prob MVN (torch) vs Log prob MVN (mine)")
plt.show()