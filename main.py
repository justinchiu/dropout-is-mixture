
import torch as th
import torch.nn as nn

from functools import reduce


# DGP:
#   * Choose z | x ~ Cat(f(x))
#   * Choose y | z ~ N(mu_z, sigma)

class Dgp(nn.Module):
    def __init__(self, dim, K):
        super(Dgp, self).__init__()
        self.dim = dim
        self.K = K

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
        N = x.shape[0]
        K = self.K
        H = self.dim
        logp_z_x = self.logp_z_x(x)
        logp_y_z = th.distributions.MultivariateNormal(
            *self.logp_y_zx(
                th.arange(0, self.K, device=x.device).unsqueeze(0).repeat(N, 1),
                x,
            )
        )
        return (
            logp_y_z.log_prob(y.view(N, 1, H).expand(N, K, H))
            + logp_z_x
        ).logsumexp(-1)

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
    def __init__(self, dim, model_dim, num_layers=3, dp=0.5):
        super(Mlp, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(dim, model_dim)] + [
            nn.Linear(model_dim, model_dim) for _ in range(num_layers - 2)
        ] + [nn.Linear(model_dim, dim)])
        self.dropout = nn.Dropout(dp)
        self.sigma = nn.Parameter(th.eye(dim))
        self.sigma.requires_grad = False

    def forward(self, x):
        return reduce(
            lambda hidden, module: self.dropout(module(hidden)),
            self.layers[1:],
            self.layers[0](x)
        ), self.sigma

def main():
    device = th.device("cuda:1")
    dim = 256
    K = 64
    N = 32
    dgp = Dgp(dim, K)
    dgp.to(device)

    dgp_hat = Dgp(dim, int(K * 1))
    dgp_hat.to(device)
    optimizer_dgp_hat = th.optim.Adam(dgp_hat.parameters(), lr = 1e-3)

    dgp_hat_op = Dgp(dim, int(K * 2))
    dgp_hat_op.to(device)
    optimizer_dgp_hat_op = th.optim.Adam(dgp_hat_op.parameters(), lr = 1e-3)

    #model_dim = dim * K // 2
    model_dim = dim * 8
    #num_layers = 3
    num_layers = 6

    #mlp = Mlp(dim, 3, 0.3)
    mlp = Mlp(dim, model_dim, num_layers, 0.)
    mlp.to(device)
    parameters = list(mlp.parameters())
    optimizer_mlp = th.optim.Adam(parameters, lr = 1e-3)

    mlp_dp = Mlp(dim, model_dim, num_layers, 0.3)
    mlp_dp.to(device)
    optimizer_mlp_dp = th.optim.Adam(mlp_dp.parameters(), lr = 1e-3)

    n_batches = int(1e3)
    for i in range(n_batches):
        x, y, z = dgp.sample((N,))

        dgp_logp_y_x = dgp.logp_y_x(y, x)
        true_nll = -dgp_logp_y_x.mean()

        dgp_hat_logp_y_x = dgp_hat.logp_y_x(y, x)
        hat_nll = -dgp_hat_logp_y_x.mean()
        optimizer_dgp_hat.zero_grad()
        hat_nll.backward()
        optimizer_dgp_hat.step()

        dgp_hat_op_logp_y_x = dgp_hat_op.logp_y_x(y, x)
        hat_op_nll = -dgp_hat_op_logp_y_x.mean()
        optimizer_dgp_hat_op.zero_grad()
        hat_op_nll.backward()
        optimizer_dgp_hat_op.step()

        mlp_dist = th.distributions.MultivariateNormal(*mlp(x))
        mlp_logp_y_x = mlp_dist.log_prob(y)
        mlp_nll = -mlp_logp_y_x.mean()
        optimizer_mlp.zero_grad()
        mlp_nll.backward()
        optimizer_mlp.step()

        mlp_dp_dist = th.distributions.MultivariateNormal(*mlp_dp(x))
        mlp_dp_logp_y_x = mlp_dp_dist.log_prob(y)
        mlp_dp_nll= -mlp_dp_logp_y_x.mean()
        optimizer_mlp_dp.zero_grad()
        mlp_dp_nll.backward()
        optimizer_mlp_dp.step()

        print(f"{mlp_dp_nll.item():.2f}, {mlp_nll.item():.2f}, {hat_op_nll.item():.2f}, {hat_nll.item():.2f}, {true_nll.item():.2f}")


if __name__ == "__main__":
    main()
