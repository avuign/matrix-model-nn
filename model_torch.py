import torch
import torch.nn as nn

from config import HIDDEN_DIM, N_LAYERS


class MatrixNetwork(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=N_LAYERS):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)
