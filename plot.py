import math

import matplotlib.pyplot as plt
import torch

from config import *
from model import MatrixNetwork


def plot(model, grid, args, save_path=None):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 13,
        }
    )

    # Learned density
    f = model(grid)
    rho = torch.sqrt(torch.relu(f))
    Z = torch.trapezoid(rho, grid, dim=0)
    rho = rho / Z

    # analytic results
    if args.m > 0:
        g = args.g4
        a = math.sqrt((math.sqrt(1 + 48 * g) - 1) / (6 * g)) if g > 0 else 2.0
        rho_analytic = (
            torch.sqrt(torch.clamp(a**2 - grid**2, min=0))
            * (2 * g * grid**2 + 0.5 + g * a**2)
        ) / torch.pi

    if args.m < 0:
        g = args.g4
        b1 = math.sqrt(1 / (4 * g) - 1 / math.sqrt(g))
        b2 = math.sqrt(1 / (4 * g) + 1 / math.sqrt(g))
        rho_analytic = (
            torch.sqrt(torch.clamp(b2**2 - grid**2, min=0))
            * torch.sqrt(torch.clamp(grid**2 - b1**2, min=0))
            * (2 * g * torch.abs(grid))
        ) / torch.pi

    print(grid[-1])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    ax.plot(
        grid.detach().numpy(),
        rho.detach().numpy(),
        color="#d62728",
        linewidth=2.2,
        label="Neural network",
    )

    if FILLING_FRAC == 0.5:
        ax.plot(
            grid.detach().numpy(),
            rho_analytic.detach().numpy(),
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Analytic",
        )

    ax.set_xlabel(r"$\lambda$", fontsize=15)
    ax.set_ylabel(r"$\rho(\lambda)$", fontsize=15)
    if args.g4 > 0:
        ax.set_title(
            rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2 + {args.g4}\,\lambda^4$",
            fontsize=15,
        )
    if args.g4 == 0.0:
        ax.set_title(rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2 $", fontsize=15)

    if FILLING_FRAC != 0.5:
        ax.set_title(
            rf"$V(\lambda) = \frac{{{args.m}}}{{2}}\lambda^2, \quad \nu={FILLING_FRAC} $",
            fontsize=15,
        )
    ax.legend(frameon=False, fontsize=12)
    ax.set_xlim(1.1 * grid[0].item(), 1.1 * grid[-1].item())
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    args = get_args()

    model = MatrixNetwork()
    print(f"loading " + WEIGHT_PATH)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    grid = args.L * torch.linspace(-1, 1, GRID_PTS)

    plot(model, grid, args, save_path=FIG_PATH)
