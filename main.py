import math

import matplotlib.pyplot as plt
import torch

from config import *
from model import MatrixNetwork


def plot(model, grid, g, save_path=None):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 13,
        }
    )

    # Learned density
    f = model(grid)
    Z = torch.trapezoid(torch.exp(f), grid, dim=0)
    rho = torch.exp(f) / Z

    # Analytic: one-cut solution
    a = math.sqrt((math.sqrt(1 + 48 * g) - 1) / (6 * g)) if g > 0 else 2.0
    rho_1cut = (
        torch.sqrt(torch.clamp(a**2 - grid**2, min=0))
        * (2 * g * grid**2 + 0.5 + g * a**2)
    ) / torch.pi

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(
        grid.numpy(),
        rho.detach().numpy(),
        color="#d62728",
        linewidth=2.2,
        label="Neural network",
    )
    ax.plot(
        grid.numpy(),
        rho_1cut.numpy(),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Analytic",
    )

    # Two-cut solution
    if g > 0 and 1 / (4 * g) > 1 / math.sqrt(g):
        b1 = math.sqrt(1 / (4 * g) - 1 / math.sqrt(g))
        b2 = math.sqrt(1 / (4 * g) + 1 / math.sqrt(g))
        rho_2cut = (
            torch.sqrt(torch.clamp(b2**2 - grid**2, min=0))
            * torch.sqrt(torch.clamp(grid**2 - b1**2, min=0))
            * (2 * g * torch.abs(grid))
        ) / torch.pi
        ax.plot(
            grid.numpy(),
            rho_2cut.numpy(),
            color="#1f77b4",
            linestyle="-.",
            linewidth=1.4,
            label="Analytic (two-cut)",
        )

    ax.set_xlabel(r"$\lambda$", fontsize=15)
    ax.set_ylabel(r"$\rho(\lambda)$", fontsize=15)
    ax.set_title(
        rf"$V(\lambda) = \frac{{1}}{{2}}\lambda^2 + {g}\,\lambda^4$", fontsize=15
    )
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(grid[0].item(), grid[-1].item())
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    model = MatrixNetwork()
    model.load_state_dict(torch.load("matrix_model_weights_gaussian.pt"))

    grid = GRID

    plot(model, grid, QUARTIC_COUPLING, save_path="./gaussian.png")
