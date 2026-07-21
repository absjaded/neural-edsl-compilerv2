import torch
import torch.nn.functional as F

def neuroflow_softclip_loss(z_fmri: torch.Tensor, z_edsl: torch.Tensor, flow_net: torch.nn.Module, tau: float = 0.07) -> torch.Tensor:
    """
    Computes the NeuroFlow + SoftCLIP loss for zero-label latent correspondence.
    z_fmri: continuous fMRI state from TACIT [batch, latent_dim]
    z_edsl: continuous vector mapped from Lean 4 DSL [batch, latent_dim]
    flow_net: neural network v_theta(x_t, t)
    """
    batch_size = z_fmri.size(0)
    device = z_fmri.device
    
    # 1. Reversible Flow Matching (RFM) Loss
    t = torch.rand(batch_size, 1).to(device)
    x_t = t * z_edsl + (1 - t) * z_fmri
    u_t_target = z_edsl - z_fmri
    
    v_pred = flow_net(x_t, t)
    loss_rfm = F.mse_loss(v_pred, u_t_target)
    
    # 2. SoftCLIP Contrastive Loss
    z_fmri_norm = F.normalize(z_fmri, dim=-1)
    z_edsl_norm = F.normalize(z_edsl, dim=-1)
    
    logits = (z_fmri_norm @ z_edsl_norm.T) / tau
    labels = torch.arange(batch_size, device=device)
    
    loss_clip = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    
    # Bidirectional consistency implicitly enforced by reversible ODE target
    total_loss = loss_rfm + 0.5 * loss_clip
    return total_loss
