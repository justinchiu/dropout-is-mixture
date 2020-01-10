
import torch as th
import torch.nn as nn

from functools import reduce


# DGP:
#   * Choose z | x ~ Cat(f(x))
#   * Choose y | z ~ N(mu_z, sigma)

class Dgp(nn.Module):
    def __init__(self, dim, K):
        super(Dgp, self).__init__()

        self.mu_x = nn.Parameter(th.randn(dim))
        self.sigma_x = nn.Parameter(th.eye(dim))
        self.z_proj = nn.Linear(dim, K, bias=False)
        # different means but shared isotropic cov
        self.mu_y = nn.Parameter(th.randn(K, dim))
        self.sigma_y = nn.Parameter(th.eye(dim))

    def logp_x(self):
        return self.mu_x, self.sigma_x

    def logp_z_x(self, x):
        return self.z_proj(x).log_softmax(-1)

    def logp_y_zx(self, z, x):
        return self.mu_y[z], self.sigma_y

    def logp_y_x(self, y, x):
        logp_z_x = self.logp_z_x(x)
        logp_y_z = th.distributions.MultivariateNormal(
            *self.logp_y_zx(self.mu_y, self.sigma_y)
        )
        return (logp_y_z.log_prob(y) + logp_z_x).logsumexp(-1)

    # p(y | x)
    def forward(self, x):
        pass

    def sample(self, shape=()):
        x = th.distributions.MultivariateNormal(*self.logp_x()).sample(shape)
        z = th.distributions.Categorical(self.logp_z_x(x)).sample()
        y = th.distributions.MultivariateNormal(*self.logp_y_zx(z, x)).sample()
        return x, y, z


# models: MLPs...ReLU i guess. predict means

class Mlp(nn.Module):
    def __init__(self, num_layers=3, dp=0.5):
        super(Mlp, self).__init__()

        self.layers = [
            nn.Linear(dim, dim) for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dp)
        self.sigma = nn.Parameter(th.eye(dim))

    def forward(self, x):
        return reduce(
            lambda hidden, module: module(self.dropout(hidden)),
            self.layers,
        ), self.sigma

def main():
    dim = 256
    K = 64
    dgp = Dgp(dim, K)
    x, y, z = dgp.sample()
    x1, y1, z1 = dgp.sample((10,))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
