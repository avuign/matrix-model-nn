import flax.linen as nn

from config import HIDDEN_DIM, N_LAYERS


class MatrixNetwork(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = N_LAYERS

    @nn.compact
    def __call__(self, x):
        x = x[..., None]
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x[..., 0]
