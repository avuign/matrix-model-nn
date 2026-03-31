import torch

GRID = torch.linspace(-5, 5, 500)


def V(x):
    return x**2 / 2
