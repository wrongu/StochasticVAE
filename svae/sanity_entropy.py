from entropy import entropy_singh_2003
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
import torch

batch = 250
samples = 8
knn = 4
d = 20
n_components = 10
reference_distribution = MixtureSameFamily(
    Categorical(torch.ones(n_components)),
    MultivariateNormal(
        loc=torch.randn(n_components, d) * 5,
        covariance_matrix=torch.eye(d).expand(n_components, d, d),
    ),
)

z = reference_distribution.sample((batch, samples))
assert z.shape == (batch, samples, d)

#%%
entropy_singh = entropy_singh_2003(
    z,
    k=knn,
    dim_samples=1,
    dim_features=-1,
)
entropy_singh_avg = entropy_singh.mean()

entropy_numeric = -reference_distribution.log_prob(z).mean(dim=1)
entropy_numeric_avg = entropy_numeric.mean()

#%%

import matplotlib.pyplot as plt

plt.plot([0, 50], [0, 50], color="red", linestyle="--")
plt.scatter(entropy_numeric, entropy_singh, marker='.')
plt.plot(entropy_numeric_avg, entropy_singh_avg, 'ok')
plt.xlabel("Entropy (Numeric)")
plt.ylabel("Entropy (Singh 2003)")
plt.title("Entropy Comparison")
plt.axis("equal")
plt.show()