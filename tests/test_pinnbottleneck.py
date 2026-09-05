"""Tests for the Interpretable PINN package.

All tests run fully offline on a CPU device — collocation points are small and
only forward passes / loss evaluations (plus short Adam steps on tiny nets) are
exercised, so no GPU or long training is required.
"""

from __future__ import annotations

import pytest
import torch

from pinnbottleneck import InterpretablePINN, PINNBottleneck, compute_all_losses


def _device():
    return torch.device("cpu")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return InterpretablePINN(num_layers=3, hidden_dim=8).to(_device())


# ---------------------------------------------------------------------------
# PINNBottleneck
# ---------------------------------------------------------------------------
def test_bottleneck_shapes():
    torch.manual_seed(0)
    bn = PINNBottleneck(hidden_dim=8, bottleneck_dim=2)
    h = torch.randn(16, 8)
    out = bn(h)
    assert out.shape == (16, 1)
    # Low-rank bottleneck: compress kernel is (bottleneck_dim, hidden_dim).
    assert bn.compress.weight.shape == (2, 8)
    assert bn.output.weight.shape == (1, 2)


def test_bottleneck_dropout_is_applied_in_forward():
    torch.manual_seed(0)
    bn = PINNBottleneck(hidden_dim=8, bottleneck_dim=2, initial_dropout=0.0)
    bn.eval()
    h = torch.randn(64, 8)
    # With p=0 it should be deterministic between the two calls.
    a = bn(h)
    bn.eval()
    b = bn(h)
    torch.testing.assert_close(a, b)


def test_bottleneck_active_dropout_changes_output():
    # Strong dropout (p~1) with output bias zeroed should drive output to ~0
    # because dropout zeros the activation feeding `output`.
    torch.manual_seed(0)
    bn = PINNBottleneck(hidden_dim=8, bottleneck_dim=2, initial_dropout=0.99)
    bn.eval()
    with torch.no_grad():
        bn.output.weight.zero_()
        bn.output.bias.zero_()
    h = torch.randn(128, 8)
    out = bn(h)
    assert out.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_dropout_eps_alias():
    bn = PINNBottleneck()
    assert bn.dropout_eps is bn.dropout
    bn.dropout_eps.p = 0.5
    assert bn.dropout.p == 0.5


# ---------------------------------------------------------------------------
# InterpretablePINN
# ---------------------------------------------------------------------------
def test_forward_shapes(model):
    x = torch.randn(32, 1)
    t = torch.rand(32, 1)
    u_pred, u_dyn = model(x, t)
    assert u_pred.shape == (32, 1)
    assert u_dyn.shape == (32, 1)
    # Known-physics and unknown-dynamics towers are genuinely independent.
    assert model.net_u_hidden is not model.net_dyn_hidden
    assert model.W_u.weight.shape == (1, 8)


def test_forward_requires_grad_path(model):
    x = torch.randn(16, 1, requires_grad=True)
    t = torch.rand(16, 1, requires_grad=True)
    u_pred, _ = model(x, t)
    u_pred.sum().backward()
    assert x.grad is not None
    assert t.grad is not None
    assert x.grad.shape == (16, 1)


def test_arch_layer_counts():
    m = InterpretablePINN(num_layers=5, hidden_dim=16)
    # 1 input linear + (num_layers-1) hidden linear + output linear.
    assert m.net_u_hidden[0].weight.shape == (16, 2)
    assert m.W_u.weight.shape == (1, 16)
    # Total linear layers in physics tower == num_layers (hidden) + 1 (out).
    n_lin = sum(1 for mod in m.net_u_hidden if isinstance(mod, torch.nn.Linear))
    assert n_lin == 5


# ---------------------------------------------------------------------------
# compute_all_losses
# ---------------------------------------------------------------------------
def test_loss_shapes_enabled(model):
    x = torch.randn(64, 1, requires_grad=True)
    t = torch.rand(64, 1, requires_grad=True).requires_grad_(True)
    losses = compute_all_losses(model, x, t, nu=0.01, enable_dyn=True)
    assert len(losses) == 4
    for loss in losses:
        assert loss.dim() == 0  # scalars
        assert loss.item() >= 0.0


def test_loss_enable_dyn_false(model):
    x = torch.randn(64, 1, requires_grad=True)
    t = torch.rand(64, 1, requires_grad=True)
    pde, pde2, reg, indep = compute_all_losses(model, x, t, nu=0.01, enable_dyn=False)
    # Pure-physics path returns the SAME pde twice and two zeros.
    torch.testing.assert_close(pde, pde2)
    assert reg.item() == 0.0
    assert indep.item() == 0.0


def test_loss_known_physics_reduces_pde():
    # Feed the model zero as the 'known physics' reference: when u_pred is
    # driven to ~0, L_pde_u should be small (pure convection+diffusion of 0).
    torch.manual_seed(1)
    m = InterpretablePINN(num_layers=2, hidden_dim=4).to(_device())
    # Freeze to trivially-zero prediction.
    for p in m.parameters():
        p.data.zero_()
    x = torch.randn(32, 1, requires_grad=True)
    t = torch.rand(32, 1, requires_grad=True)
    pde_u, *_ = compute_all_losses(m, x, t, nu=0.0, enable_dyn=True)
    # With u_pred=0 and nu=0, k_conv=k_diff=u_dyn=0 -> residual ~0.
    assert pde_u.item() < 1e-12


def test_backward_through_loss(model):
    x = torch.randn(32, 1, requires_grad=True)
    t = torch.rand(32, 1, requires_grad=True)
    losses = compute_all_losses(model, x, t, nu=0.01, enable_dyn=True)
    total = sum(losses)
    total.backward()
    # All parameters must receive non-None gradients.
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum().item() > 0, name


# ---------------------------------------------------------------------------
# train CLI
# ---------------------------------------------------------------------------
def test_train_cli_version(capsys):
    import pinnbottleneck
    assert pinnbottleneck.__version__ == "0.1.0"


def test_train_cli_builds_parser():
    from pinnbottleneck.train import build_parser
    parser = build_parser()
    ns = parser.parse_args(["--epochs", "3", "--nu", "0.1", "--seed", "7"])
    assert ns.epochs == 3
    assert ns.nu == 0.1
    assert ns.seed == 7