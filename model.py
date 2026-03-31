import torch
import torch.nn as nn


class MatrixNetwork(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)
