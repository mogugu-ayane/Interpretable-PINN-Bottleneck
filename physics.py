import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

def compute_all_losses(model, x, t, nu, lambda_reg=1.0, beta_indep=1.0, eps_safe=1e-4, enable_dyn=True):
    
    u_pred, u_dyn = model(x, t)
    u_t = autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    u_x = autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]

    k_conv = -u_pred * u_x
    k_diff = nu * u_xx

    if not enable_dyn:
        u_t_assemble = k_conv + k_diff
        loss_pde = torch.mean((u_t - u_t_assemble)**2)
        zero = torch.tensor(0.0, device=x.device)
        return loss_pde, loss_pde, zero, zero
    else:
        u_t_assemble_u = k_conv + k_diff + u_dyn.detach()
        loss_pde_u = torch.mean((u_t - u_t_assemble_u)**2)

        u_t_assemble_dyn = k_conv.detach() + k_diff.detach() + u_dyn
        loss_pde_dyn = torch.mean((u_t.detach() - u_t_assemble_dyn)**2)

        u_dyn_sq = u_dyn**2
        loss_u_reg = lambda_reg * torch.mean(u_dyn_sq / (u_dyn_sq + eps_safe))

        inner_product_sq = (k_conv.detach() * u_dyn)**2
        loss_indep = beta_indep * torch.mean(inner_product_sq / (inner_product_sq + eps_safe))

        return loss_pde_u, loss_pde_dyn, loss_u_reg, loss_indep
