import pickle

import jax
import jax.numpy as jnp
import optax

from config import *
from model_jax import MatrixNetwork


def compute_loss(params, model, V, grid, nu=0.5):

    rho = model.apply({"params": params}, grid)
    Z = jnp.trapezoid(rho, grid)
    rho = rho / Z

    potential_term = jnp.trapezoid(V(grid) * rho, grid)

    xi, yi = jnp.meshgrid(grid, grid, indexing="ij")
    diff = jnp.abs(xi - yi)

    mask = ~jnp.eye(len(grid), dtype=bool)
    diff = jnp.where(mask, diff, 1.0)

    log_matrix = jnp.log(diff) * mask

    rho_matrix = jnp.outer(rho, rho)

    integrand = rho_matrix * log_matrix

    kernel_term = jnp.trapezoid(jnp.trapezoid(integrand, grid), grid)

    loss = potential_term - kernel_term

    return loss


def train(params, model, V, grid, num_epochs, lr):

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(compute_loss)(params, model, V, grid)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state)
        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}, Loss: {loss}")
    return params


if __name__ == "__main__":
    args = get_args()
    model = MatrixNetwork()

    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones(1))["params"]

    grid = args.L * jnp.linspace(-1, 1, GRID_PTS)

    def V(x):
        return (
            (args.m) * (x**2) / 2
            + args.g4 * (x**4)
            + args.g6 * (x**6)
            + args.g8 * (x**8)
            + args.g10 * (x**10)
        )

    print("Starting to train..")
    params = train(params, model, V, grid, args.n_epochs, args.lr)

    print(f"Neural network trained. Saving weights at " + WEIGHT_PATH_JAX)

    with open(WEIGHT_PATH_JAX, "wb") as f:
        pickle.dump(params, f)
