# Neural Network for Matrix Models

A neural network approach to compute the large $N$ eigenvalue density of Hermitian one-matrix models. Two independent implementations are provided: **JAX/Flax/Optax** and **PyTorch**.

## The Idea

In the large $N$ limit of a Hermitian one-matrix model with potential $V(\lambda)$, the eigenvalue density minimizes the action functional

$$S[\rho] = N^2 \left[\int V(\lambda) \rho(\lambda) d\lambda - \iint \log|\lambda - \mu| \rho(\lambda) \rho(\mu) d\lambda d\mu\right]$$

subject to $\rho \geq 0$ and $\int \rho = 1$.

The density is represented by a neural network $f_\theta\colon \mathbb{R} \to \mathbb{R}$, normalized to satisfy both constraints by construction. The action is discretized on a uniform grid via the trapezoidal rule and minimized with standard backpropagation.

## Quick Start

### JAX (default)

```bash
pip install jax jaxlib flax optax matplotlib

# Train on the Gaussian potential (recovers the Wigner semicircle)
python train_jax.py

# Plot
python plot.py --model jax
```

### PyTorch

```bash
pip install torch matplotlib

# Train
python train_torch.py

# Plot
python plot.py --model torch
```

Both training scripts accept the same CLI flags. See [Usage](#usage) for the full reference.

---

## Results

### Gaussian potential: $V(\lambda) = \frac{1}{2}\lambda^2$

Recovers the Wigner semicircle $\rho(\lambda) = \frac{2}{\pi}\sqrt{1 - \lambda^2}$.

![Gaussian](images/m_1.0_g4_0.0.png)

### Quartic potential (one-cut): $V(\lambda) = \frac{1}{2}\lambda^2 + g\lambda^4$

For positive mass and quartic coupling, the saddle point is a deformed semicircle supported on a single interval.

![One-cut quartic](images/m_1.0_g4_3.0.png)

### Quartic potential (two-cut): $V(\lambda) = -\frac{1}{2}\lambda^2 + g\lambda^4$

Negative mass triggers a $\mathbb{Z}_2$-symmetric two-cut phase. The network discovers the disconnected support from the action alone.

![Two-cut quartic](images/m_-1.0_g4_0.03.png)

### Filling fractions

For the two-cut solution, imposing $\nu = \int_0^\infty \rho\,d\lambda \neq \frac{1}{2}$ selects $\mathbb{Z}_2$-breaking saddle points, enforced via a penalty term in the loss. Example: $\nu = 0.2$ (PyTorch only).

![Two-cut with filling fraction](images/m_-1.0_g4_0.02.png)

---

## Architecture

Both backends implement a fully connected MLP:

| | JAX (`model_jax.py`) | PyTorch (`model_torch.py`) |
|---|---|---|
| Activation | GELU | Tanh |
| Output nonlinearity | ReLU | — |
| Density ansatz | $\text{relu}(f_\theta) / Z$ | $e^{f_\theta} / Z$ |
| Hidden dim | 128 | 128 |
| Layers | 6 | 6 |

In both cases the normalization $Z = \int \rho_\theta\,d\lambda$ is computed on the training grid via the trapezoidal rule, so positivity and unit normalization are enforced exactly.

---

## Usage

All flags are defined in `config.py` and shared across `train_jax.py`, `train_torch.py`, and `plot.py`.

### Potential

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--m` | float | `1.0` | Coefficient of $\frac{1}{2}\lambda^2$. Negative values trigger a two-cut phase. |
| `--g4` | float | `0.0` | Quartic coupling $g_4\lambda^4$. |
| `--g6` | float | `0.0` | Sextic coupling $g_6\lambda^6$. |
| `--g8` | float | `0.0` | Octic coupling $g_8\lambda^8$. |

The full potential is $V(\lambda) = \frac{m}{2}\lambda^2 + g_4\lambda^4 + g_6\lambda^6 + g_8\lambda^8$.

### Training

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_epochs` | int | `4000` | Number of gradient steps. |
| `--lr` | float | `0.001` | Adam learning rate. |
| `--L` | float | `2.0` | Grid half-width: training and plotting grid is $[-L, L]$ with 800 points. Should comfortably contain the support. |

### Physics

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--nu` | float | `0.5` | Target filling fraction $\nu = \int_0^\infty \rho\,d\lambda$. Default `0.5` is the $\mathbb{Z}_2$-symmetric saddle. Any other value activates a quadratic penalty $5(\nu_\theta - \nu)^2$ in the PyTorch loss. **Not implemented in the JAX training loop.** |

### Plotting

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `"jax"` | Backend to load weights from: `"jax"` or `"torch"`. Used by `plot.py`. |
| `--plot_an` | bool | `False` | Overlay the exact analytic solution. Supported for the quartic potential ($g_4 > 0$, any sign of $m$) only. |

### Saved files

Weights and figures are named automatically from the potential parameters:

```
saved_weights/m_{m}_g4_{g4}.pt     # PyTorch weights
weights_jax/m_{m}_g4_{g4}.pkl      # JAX weights
images/m_{m}_g4_{g4}.png           # Output figure
```

### Examples

```bash
# Gaussian potential (Wigner semicircle), JAX
python train_jax.py
python plot.py --model jax --plot_an True

# One-cut quartic, PyTorch
python train_torch.py --g4 3
python plot.py --model torch --g4 3 --plot_an True

# Two-cut phase (wider grid needed)
python train_jax.py --m -1 --g4 0.03 --L 5
python plot.py --model jax --m -1 --g4 0.03 --L 5 --plot_an True

# Two-cut with asymmetric filling fraction (PyTorch only)
python train_torch.py --m -1 --g4 0.02 --L 5 --nu 0.2
python plot.py --model torch --m -1 --g4 0.02 --L 5 --nu 0.2

# Sextic potential
python train_jax.py --m 1 --g6 0.5 --L 3

# Longer training
python train_torch.py --g4 3 --n_epochs 10000 --lr 5e-4
```

---

## References

- E. Brézin, C. Itzykson, G. Parisi, J.-B. Zuber, *Planar Diagrams*, Commun. Math. Phys. **59** (1978) 35–51
- P. Di Francesco, P. Ginsparg, J. Zinn-Justin, *2D Gravity and Random Matrices*, Phys. Rept. **254** (1995) 1–133, [arXiv:hep-th/9306153](https://arxiv.org/abs/hep-th/9306153)
