"""Physics-informed loss terms for the Interpretable PINN.

Computes the Burgers'-equation residual losses that disentangle the known
physics (convection ``k_conv``, diffusion ``k_diff``) from the unmodeled
dynamics channel ``u_dyn``, plus the steering/capacity penalties:

* ``L_u_reg`` — regularizer that keeps the unknown-dynamics term sparse;
* ``L_indep`` — an orthogonality penalty that forces ``u_dyn`` to be
  independent of the frozen known-physics term (ideally, unknown dynamics
  must explain what physics cannot, without the physics exploiting it).
"""

from __future__ import annotations

import torch
import torch.autograd as autograd
from torch import nn  # noqa: F401  (public re-export for callers)

from .pinn import InterpretablePINN


def compute_all_losses(
    model: InterpretablePINN,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float,
    lambda_reg: float = 1.0,
    beta_indep: float = 1.0,
    eps_safe: float = 1e-4,
    enable_dyn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the four PINN loss components for collocation points ``(x, t)``.

    Parameters
    ----------
    model:
        The :class:`InterpretablePINN` instance being trained.
    x:
        Collocation spatial coordinates, shape ``(batch, 1)`` (requires grad).
    t:
        Collocation time coordinates, shape ``(batch, 1)`` (requires grad).
    nu:
        Kinematic viscosity for the Burgers' diffusion term.
    lambda_reg:
        Weight for the unknown-dynamics sparsity regularizer ``L_u_reg``.
    beta_indep:
        Weight for the physics/dynamics independence penalty ``L_indep``.
    eps_safe:
        Small positive constant guarding the bounded-normalisation
        denominators from division by zero.
    enable_dyn:
        When ``False``, the dynamic channel is disabled entirely and the
        function returns ``(loss_pde, loss_pde, 0, 0)`` (the pure-physics
        reference case).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(L_pde_u, L_pde_dyn, L_u_reg, L_indep)`` where:
            L_pde_u  — PDE residual measured against predicted dynamics;
            L_pde_dyn— PDE residual measured against the dynamic channel;
            L_u_reg  — sparsity regularizer on ``u_dyn``;
            L_indep  — orthogonality penalty between physics and dynamics.
    """
    u_pred, u_dyn = model(x, t)

    # First and second spatial/temporal derivatives of the physics channel.
    u_t = autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    u_x = autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]

    # Known physics terms (Burgers: convection + diffusion).
    k_conv = -u_pred * u_x
    k_diff = nu * u_xx

    if not enable_dyn:
        u_t_assemble = k_conv + k_diff
        loss_pde = torch.mean((u_t - u_t_assemble) ** 2)
        zero = torch.tensor(0.0, device=x.device)
        return loss_pde, loss_pde, zero, zero

    # PDE residual against the predicted dynamics channel.
    u_t_assemble_u = k_conv + k_diff + u_dyn.detach()
    loss_pde_u = torch.mean((u_t - u_t_assemble_u) ** 2)

    # PDE residual against the dynamic channel (physics detached).
    u_t_assemble_dyn = k_conv.detach() + k_diff.detach() + u_dyn
    loss_pde_dyn = torch.mean((u_t.detach() - u_t_assemble_dyn) ** 2)

    # Bounded sparsity regularizer on the unknown-dynamics magnitude.
    u_dyn_sq = u_dyn**2
    loss_u_reg = lambda_reg * torch.mean(u_dyn_sq / (u_dyn_sq + eps_safe))

    # Bounded orthogonality penalty |k_hat * u_hat|^2 / (|k_hat*u_hat|^2 + 1)
    inner_product_sq = (k_conv.detach() * u_dyn) ** 2
    loss_indep = beta_indep * torch.mean(
        inner_product_sq / (inner_product_sq + eps_safe)
    )

    return loss_pde_u, loss_pde_dyn, loss_u_reg, loss_indep