def compute_all_losses(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u, concept_preds, loss_indep, loss_rec = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    f = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(f**2)
    
    y_conv, y_diff = model.bottleneck.generate_gt_labels(u, u_x, u_xx, nu)
    gt_concepts = torch.cat([y_conv, y_diff], dim=1).detach() 
    loss_concept = nn.MSELoss()(concept_preds, gt_concepts)
    
    return loss_pde, loss_concept, loss_indep, loss_rec
