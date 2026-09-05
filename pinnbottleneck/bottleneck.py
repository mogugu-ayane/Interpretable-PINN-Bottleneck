"""Low-rank concept-bottleneck head for the unknown-dynamics channel.

This module is the cleaned-up home of the :class:`PINNBottleneck` module that
previously lived in a stray, extension-less ``models`` file.  It extracts the
*unmodeled dynamics* channel ``u_dyn`` from the shared features ``h`` under an
extremely constrained (low-rank) capacity so the network cannot "cheat" by
folding residual error into the known-physics prediction ``u_pred``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PINNBottleneck(nn.Module):
    """Low-rank bottleneck that isolates the unknown-dynamics contribution.

    The module warps the shared hidden state ``h`` down to a very small
    ``bottleneck_dim`` (a hard form of memorisation/cheating prevention),
    pushes it through a C2-smooth activation (Tanh is used so that the
    second-order derivatives required by the PINN objective remain finite),
    applies optional residual dropout, and finally maps down to a scalar
    forcing term ``u_dyn``.

    Parameters
    ----------
    hidden_dim:
        Width of the incoming feature vector ``h`` (must match the dense head
        that feeds this module).
    bottleneck_dim:
        Dimensionality of the information bottleneck.  Small values (e.g. 2)
        forcibly cap how much unrelated signal can be smuggled into the
        unknown-dynamics channel.
    initial_dropout:
        Dropout probability on the residual channel.  Exposed as an attribute
        (``dropout_eps``) so training loops may change ``p`` mid-run.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        bottleneck_dim: int = 2,
        initial_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. Extreme dimensionality reduction (hidden_dim -> 2).
        self.compress = nn.Linear(hidden_dim, bottleneck_dim)
        # 2. C2-continuous activation (Tanh/SiLU required for PINN 2nd derivs).
        self.activation = nn.Tanh()
        # 3. Residual dropout (`dropout_eps`) - discourages reliance on the
        #    unknown-dynamics channel.  Applied in forward() so mid-training
        #    scheduling of `p` (see `dropout_eps`) actually takes effect.
        self.dropout = nn.Dropout(p=initial_dropout)
        self.dropout_eps = self.dropout  # scheduling alias used by scripts
        # 4. Final scalar output (bottleneck_dim -> 1).
        self.output = nn.Linear(bottleneck_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map shared features ``h`` to the scalar unknown-dynamics term.

        Parameters
        ----------
        h:
            Shape ``(batch, hidden_dim)`` — features produced by a shared
            encoder for the current collocation point.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` residual/unknown-dynamics forcing ``u_dyn``.
        """
        x = self.compress(h)      # physical capacity removal (W_u1)
        x = self.activation(x)    # non-linearity
        x = self.dropout(x)       # residual dropout (train-time only)
        u_dyn = self.output(x)    # final error-residual output (W_u2)
        return u_dyn