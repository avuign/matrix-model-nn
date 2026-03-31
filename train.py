import torch
import torch.nn as nn

from config import *
from model import MatrixNetwork


def compute_loss(model, V, grid):

    f = model(grid)
    Z = torch.trapezoid(torch.exp(f), grid, dim=0)
    rho = torch.exp(f) / Z

    int1 = torch.trapezoid(V(grid) * rho, grid, dim=0)
    xi, yi = torch.meshgrid(grid, grid, indexing="ij")

    rho_matrix = torch.outer(rho, rho)
    log_matrix = torch.log(torch.abs(xi - yi))
    for i in range(len(grid)):
        log_matrix[i, i] = 0
    integrand = rho_matrix * log_matrix

    int2 = torch.trapezoid(torch.trapezoid(integrand, grid, dim=0), grid)

    return int1 - int2


def train(model, V, grid, num_epochs, lr):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        optimizer.zero_grad()
        loss = compute_loss(model, V, grid)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if epoch % 500 == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


if __name__ == "__main__":
    model = MatrixNetwork()

    grid = GRID

    print("Starting to train..")
    train(model, V, grid, 5000, 0.001)

    print("Neural network trained. Saving weights.")

    torch.save(model.state_dict(), "matrix_model_weights.pt")
