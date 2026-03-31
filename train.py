import torch
import torch.nn as nn

from config import *
from model import MatrixNetwork


def compute_loss(model, V, grid, nu=0.5):

    f = model(grid)
    rho = torch.sqrt(torch.relu(f))
    Z = torch.trapezoid(rho, grid, dim=0)
    rho = rho / Z

    potential_term = torch.trapezoid(V(grid) * rho, grid, dim=0)

    xi, yi = torch.meshgrid(grid, grid, indexing="ij")
    diff = torch.abs(xi - yi)

    mask = ~torch.eye(len(grid), dtype=torch.bool)
    diff = diff + (~mask).float()  # put 1 on diagonal so log(1) = 0

    log_matrix = torch.log(diff) * mask

    rho_matrix = torch.outer(rho, rho)

    integrand = rho_matrix * log_matrix

    kernel_term = torch.trapezoid(torch.trapezoid(integrand, grid, dim=0), grid)

    loss = potential_term - kernel_term

    if nu != 0.5:
        positive_mask = grid > 0
        filling = torch.trapezoid(rho[positive_mask], grid[positive_mask])
        loss += 5 * (filling - nu) ** 2

    return loss


def train(model, V, grid, num_epochs, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        optimizer.zero_grad()
        loss = compute_loss(model, V, grid, nu=FILLING_FRAC)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


if __name__ == "__main__":
    args = get_args()
    model = MatrixNetwork()

    grid = args.L * torch.linspace(-1, 1, GRID_PTS)

    def V(x):
        return (args.m) * (x**2) / 2 + args.g4 * (x**4)

    print("Starting to train..")
    train(model, V, grid, args.n_epochs, args.lr)

    print(f"Neural network trained. Saving weights at " + WEIGHT_PATH)

    torch.save(model.state_dict(), WEIGHT_PATH)
