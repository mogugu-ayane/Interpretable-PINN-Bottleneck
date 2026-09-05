"""Interpretable PINN with Concept Bottleneck Layers for Burgers' equation.

A physics-informed neural network that factorises the solution into a
*known-physics* channel (``u_pred``) and an *unknown-dynamics* channel
(``u_dyn``) through a low-rank concept bottleneck, enabling physical steering
and per-channel audit of residuals.
"""

from __future__ import annotations

from .bottleneck import PINNBottleneck
from .loss import compute_all_losses
from .pinn import InterpretablePINN

__all__ = ["InterpretablePINN", "PINNBottleneck", "compute_all_losses", "__version__"]

__version__ = "0.1.0"