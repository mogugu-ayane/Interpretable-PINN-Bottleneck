# Interpretable-PINN-Bottleneck

Physics-Informed Neural Networks (PINNs) with **Concept Bottleneck Layers** for
the viscous Burgers' equation: disentangle **known physics**, **unmodeled
dynamics**, and **residuals** for physical steering and faithfulness.

## What it does

A standard PINN is trained to satisfy a PDE by minimising a residual loss. The
frame often *collapses* — the network satisfies the residual algebraically
(surrogate + correction terms can cancel) without ever learning the physical
laws. This package attacks that collapse with a **concept bottleneck**:

| Channel | Symbol | Meaning |
|---|---|---|
| Known physics | `u_pred` | Pure flow learned by an undisturbed dense tower (convection + diffusion). |
| Unknown dynamics | `u_dyn` | Residual forcing extracted by a **low-rank bottleneck** (`hidden_dim -> 2 -> 1`) with limited capacity, so it cannot hide the physics' failure. |
| Residual | `ε` | Orthogonality / bounded normalisation penalties keep the channels independent. |

Two mechanisms make the channels behave independently:

* **Gradient isolation** — the physics terms are `.detach()`-ed when measuring
  the dynamic channel (and vice-versa), so auxiliary losses cannot interfere
  with the PDE learning path.
* **Bounded, non-divergent penalties** — sparsity and independence terms use
  the bounded form `a² / (a² + ε)` (clamped to `[0, 1)`) to stop the
  second-order L-BFGS curvature from blowing up.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # pytorch (CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu)
```

Requires Python >= 3.10 and `torch` + `numpy`.

## Usage

Run the turn-key Burgers' training:

```bash
python -m pinnbottleneck.train --epochs 500 --animate-output shock.png
```

CLI options:

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 500 | Adam warm-up epochs (two-phase: Adam → L-BFGS) |
| `--lbfgs-iters` | 200 | L-BFGS fine-tune iterations |
| `--nu` | `0.01/π` | Viscosity |
| `--device` | auto | `cuda` / `mps` / `cpu` |
| `--seed` | 42 | RNG seed for reproducibility |
| `--animate-output PATH` | — | Render `u(x, t=0.5)` snapshot to a file |
| `--plot-backend` | — | Force interactive pyplot (off by default; headless-safe) |

As a library:

```python
import torch
from pinnbottleneck import InterpretablePINN, compute_all_losses

model = InterpretablePINN(num_layers=4, hidden_dim=64)
x = torch.rand(128, 1, requires_grad=True)
t = torch.rand(128, 1, requires_grad=True)

u_pred, u_dyn = model(x, t)          # known physics vs. unknown dynamics
losses = compute_all_losses(model, x, t, nu=0.01)
# (L_pde_u, L_pde_dyn, L_u_reg, L_indep)
```

## Module layout

```
bottleneck.py    PINNBottleneck — low-rank (hidden_dim -> bottleneck -> 1) head
pinn.py          InterpretablePINN — dual-tower model (physics + dynamics)
loss.py          compute_all_losses — the four disentangled loss terms
train.py         CLI training entry point (Adam → L-BFGS two-phase)
tests/           13 offline CPU tests
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q            # 13 tests, fully offline on CPU
ruff check .
```

The training loop uses the two-phase schedule from the project notes — an Adam
warm-up (with a **mid-training residual-dropout bump** from `0.1` to `0.3` at
~66% of epochs to suppress residual domination) followed by L-BFGS fine-tuning.