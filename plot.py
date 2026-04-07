import math
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import *
from model import MatrixNetwork as TorchModel
from model_jax import MatrixNetwork as JaxModel


def plot_torch(model, grid, args, save_path=None):
    f = model(grid)
    rho = torch.exp(f)
    rho = rho / torch.trapezoid(rho, grid)
    plot(rho.detach().numpy(), grid.detach().numpy(), args, save_path)


def plot_jax(params, model, grid, args, save_path=None):
    f = model.apply({"params": params}, grid)
    rho = jnp.exp(f)
    rho = rho / jnp.trapezoid(rho, grid)
    plot(np.array(rho), np.array(grid), args, save_path)


def plot(rho_np, grid_np, args, save_path=None):

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 13})

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(grid_np, rho_np, color="#d62728", linewidth=2.2, label="Neural network")

    if args.plot_an:
        g = args.g4
        if args.m > 0:
            a = math.sqrt((math.sqrt(1 + 48 * g) - 1) / (6 * g)) if g > 0 else 2.0
            rho_analytic = (
                np.sqrt(np.clip(a**2 - grid_np**2, 0, None))
                * (2 * g * grid_np**2 + 0.5 + g * a**2)
            ) / np.pi
        else:
            b1 = math.sqrt(1 / (4 * g) - 1 / math.sqrt(g))
            b2 = math.sqrt(1 / (4 * g) + 1 / math.sqrt(g))
            rho_analytic = (
                np.sqrt(np.clip(b2**2 - grid_np**2, 0, None))
                * np.sqrt(np.clip(grid_np**2 - b1**2, 0, None))
                * (2 * g * np.abs(grid_np))
            ) / np.pi
        ax.plot(
            grid_np,
            rho_analytic,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Analytic",
        )

    ax.set_xlabel(r"$\lambda$", fontsize=15)
    ax.set_ylabel(r"$\rho(\lambda)$", fontsize=15)

    if FILLING_FRAC != 0.5:
        ax.set_title(
            rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2, \quad \nu={FILLING_FRAC}$",
            fontsize=15,
        )
    elif args.g4 > 0:
        ax.set_title(
            rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2 + {args.g4}\,\lambda^4$",
            fontsize=15,
        )
    else:
        ax.set_title(rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2$", fontsize=15)

    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(1.1 * grid_np[0], 1.1 * grid_np[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    params = None

    match args.model:
        case "jax":
            model = JaxModel()
            print(f"loading " + WEIGHT_PATH_JAX)
            with open(WEIGHT_PATH_JAX, "rb") as f:
                params = pickle.load(f)
            grid = args.L * jnp.linspace(-1, 1, GRID_PTS)
            plot_jax(params, model, grid, args, save_path=FIG_PATH)
        case "torch":
            model = TorchModel()
            print(f"loading " + WEIGHT_PATH)
            model.load_state_dict(torch.load(WEIGHT_PATH))
            grid = args.L * torch.linspace(-1, 1, GRID_PTS)
            plot_torch(model, grid, args, save_path=FIG_PATH)
        case _:
            print("choose one of the valid models : torch, jax")
