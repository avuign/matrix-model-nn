import flax.linen as nn
import jax
import jax.numpy as jnp

from config import HIDDEN_DIM, N_LAYERS


class MatrixNetwork(nn.Module):
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = N_LAYERS

    @nn.compact
    def __call__(self, x):
        R_raw = self.param("R_raw", lambda key, shape: jnp.array(1.0), ())
        R = nn.softplus(R_raw)

        z = x[..., None]
        for _ in range(self.num_layers):
            z = nn.Dense(self.hidden_dim)(z)
            z = nn.gelu(z)
        z = nn.Dense(1)(z)
        z = z[..., 0]

        return nn.softplus(z)
