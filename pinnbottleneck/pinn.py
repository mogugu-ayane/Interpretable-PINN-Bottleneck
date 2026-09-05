"""Interpretable Physics-Informed Neural Network (PINN) for Burgers' equation.

The model factorises the solution into two independent channels:

* ``u_pred`` — the **known-physics** prediction driven purely by the
  hand-coded Burgers' operator (convection/diffusion terms in ``loss.py``).
* ``u_dyn`` — the **unknown-dynamics** contribution captured by the
  low-rank concept bottleneck (:class:`~bottleneck.PINNBottleneck`).

This separation lets the residual forcing term be steered and audited
independently, which is the core idea of the "Interpretable PINN with Concept
Bottleneck Layers" line of work this repository explores.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bottleneck import PINNBottleneck


class InterpretablePINN(nn.Module):
    """Burgers' equation PINN with a disentangled concept bottleneck.

    Two independent towers share the same ``(x, t)`` collocation input:

    * a *physics* tower maps to ``u_pred`` through a dense Tanh MLP;
    * a *dynamics* tower maps through the low-rank :class:`PINNBottleneck` to
      the unknown forcing ``u_dyn``.

    Parameters
    ----------
    num_layers:
        Number of hidden layers in each tower (each ``hidden_dim`` wide).
    hidden_dim:
        Width of the hidden layers in both towers.
    bottleneck_dim:
        Dimensionality of the concept bottleneck used for the unknown
        dynamics channel.
    """

    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 64,
        bottleneck_dim: int = 2,
    ) -> None:
        super().__init__()

        # ---- Independent network 1: main flow (u_pred) tower ----------
        layers_u: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers_u.append(nn.Linear(hidden_dim, hidden_dim))
            layers_u.append(nn.Tanh())
        self.net_u_hidden = nn.Sequential(*layers_u)
        self.W_u = nn.Linear(hidden_dim, 1)

        # ---- Independent network 2: unknown dynamics tower ------------
        layers_dyn: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers_dyn.append(nn.Linear(hidden_dim, hidden_dim))
            layers_dyn.append(nn.Tanh())
        self.net_dyn_hidden = nn.Sequential(*layers_dyn)

        # Bottleneck module for the unknown-dynamics channel.
        self.bottleneck = PINNBottleneck(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the known-physics prediction and the unknown-dynamics term.

        The known physics ``u_pred`` is trained pure and undisturbed; the
        residual forcing ``u_dyn`` is separated from the shared skeleton and
        carries its own ``L_u_reg`` penalty (see ``loss.py``).

        Parameters
        ----------
        x:
            Spatial coordinates, shape ``(batch, 1)``.
        t:
            Time coordinates, shape ``(batch, 1)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(u_pred, u_dyn)``, each of shape ``(batch, 1)``.
        """
        inputs = torch.cat([x, t], dim=1)  # (batch, 2)

        # 1. Main flow learns pure physical law, undisturbed.
        h_u = self.net_u_hidden(inputs)
        u_pred = self.W_u(h_u)

        # 2. Unknown dynamics isolated in its own tower.
        h_dyn = self.net_dyn_hidden(inputs)
        u_dyn = self.bottleneck(h_dyn)

        return u_pred, u_dyn