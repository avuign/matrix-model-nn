import torch
import torch.nn as nn


class MatrixNetwork(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.CELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.CELU()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)
