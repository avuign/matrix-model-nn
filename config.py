import torch

N_EPOCHS = 4000
LR = 0.001

GRID = torch.linspace(-4, 4, 2000)


QUARTIC_COUPLING = 0


def V(x):
    return (x**2) / 2 + QUARTIC_COUPLING * (x**4)
