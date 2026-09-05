"""Turn-key Burgers' training script for the Interpretable PINN.

This replaces the previous broken, throwaway ``loop`` script (which referenced
dozens of undefined names, called the constructor with the wrong signature,
mis-unpacked the model output, and never set ``requires_grad`` on inputs so
autograd would have raised).  It is a real, reproducible entry point:

    python -m pinnbottleneck.train --epochs 500 --animate-output out.png

Exposes the same two-phase optimiser schedule used by the research notes
(Adam warm-up, then L-BFGS fine-tuning) behind a small argparse CLI.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe plotting
    from matplotlib import pyplot as plt
    _HAS_PLT = True
except Exception:  # pragma: no cover - matplotlib optional
    _HAS_PLT = False


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Interpretable PINN on Burgers' equation.")
    p.add_argument("--epochs", type=int, default=500, help="Adam warm-up epochs")
    p.add_argument("--lbfgs-iters", type=int, default=200, help="L-BFGS fine-tune iterations")
    p.add_argument("--nu", type=float, default=0.01 / np.pi, help="Viscosity")
    p.add_argument("--device", type=str, default=None, help="torch device (auto if omitted)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--animate-output", type=str, default=None, metavar="PATH",
                   help="Render a snapshot plot of u(x, t=0.5) to a file")
    p.add_argument("--plot-backend", action="store_true",
                   help="Force interactive pyplot (off by default; headless-safe Agg)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Collocation / IC / BC sampling (Burgers easier: 1D viscous Burgers).
    N_u = 100
    N_f = 5000

    x_f = (torch.rand(N_f, 1, device=device) * 2 - 1.0).requires_grad_(True)
    t_f = (torch.rand(N_f, 1, device=device) * 0.99).requires_grad_(True)

    x_ic = (torch.rand(N_u, 1, device=device) * 2 - 1.0)
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic.to(torch.float32))

    t_bc = torch.rand(N_u, 1, device=device) * 0.99
    x_bc = torch.where(
        torch.rand(N_u, 1, device=device) > 0.5,
        torch.ones_like(t_bc),
        -torch.ones_like(t_bc),
    )
    u_bc = torch.zeros_like(t_bc)

    from .loss import compute_all_losses
    from .pinn import InterpretablePINN

    model = InterpretablePINN(num_layers=9, hidden_dim=20).to(device)

    lambda_concept = 0.01
    lambda_indep = 0.001
    lambda_rec = 0.001
    nu = args.nu

    def _composite_loss() -> torch.Tensor:
        loss_pde, loss_concept, loss_indep, loss_rec = compute_all_losses(
            model, x_f, t_f, nu
        )
        u_ic_pred = model(x_ic, t_ic)[0]
        u_bc_pred = model(x_bc, t_bc)[0]
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)
        return (
            loss_pde
            + loss_ic
            + loss_bc
            + lambda_concept * loss_concept
            + lambda_indep * loss_indep
            + lambda_rec * loss_rec
        )

    # ---- Phase 1: Adam warm-up --------------------------------------
    print(f"[pinnbottleneck] device={device} Adam warm-up ({args.epochs} epochs)")
    opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        # Mid-training residual dropout bump (residual-domination suppression).
        if epoch == int(args.epochs * 0.66):
            model.bottleneck.dropout_eps.p = 0.3
            print(f"  -> Epoch {epoch}: residual dropout -> 0.3 (mid-training)")
        opt_adam.zero_grad()
        loss = _composite_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_adam.step()
        if (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
            print(f"Adam Epoch {epoch + 1:04d} | Total Loss: {loss.item():.4e}")

    # ---- Phase 2: L-BFGS fine-tuning --------------------------------
    print(f"[pinnbottleneck] L-BFGS fine-tuning ({args.lbfgs_iters} iters)")
    model.eval()
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=args.lbfgs_iters,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
    )

    def closure() -> torch.Tensor:
        opt_lbfgs.zero_grad()
        loss = _composite_loss()
        loss.backward()
        return loss

    opt_lbfgs.step(closure)
    final_loss = closure()
    print(f"Final Loss: {final_loss.item():.4e}")

    # ---- Optional snapshot plot --------------------------------------
    if args.animate_output:
        if not _HAS_PLT:
            print("matplotlib not available; skipping plot")
        else:
            with torch.no_grad():
                x_test = torch.linspace(-1, 1, 256).view(-1, 1).to(device)
                t_test = torch.full_like(x_test, 0.5)
                u_pred = model(x_test, t_test)[0].cpu().numpy()
            x_test_np = x_test.cpu().numpy()
            fig = plt.figure(figsize=(8, 5))
            plt.plot(x_test_np, u_pred, "b-", linewidth=2,
                     label="Interpretable PINN (t=0.5)")
            plt.title("Interpretable PINN - Shockwave Formation")
            plt.xlabel("x")
            plt.ylabel("u(x,t)")
            plt.grid(True, linestyle="--")
            plt.legend()
            fig.savefig(args.animate_output, dpi=120)
            print(f"Wrote snapshot to {args.animate_output}")
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())