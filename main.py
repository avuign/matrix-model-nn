import matplotlib.pyplot as plt
import torch

from config import *
from model import MatrixNetwork


def plot(model, grid):

    f = model(grid)
    Z = torch.trapezoid(torch.exp(f), grid, dim=0)
    rho = torch.exp(f) / Z

    rho_analytic = torch.sqrt(4 - grid**2) * 1 / (2 * torch.pi)

    plt.plot(grid.numpy(), rho.detach().numpy())
    plt.plot(grid.numpy(), rho_analytic.numpy())
    plt.show()


if __name__ == "__main__":
    model = MatrixNetwork()
    model.load_state_dict(torch.load("matrix_model_weights.pt"))

    grid = GRID

    plot(model, grid)
