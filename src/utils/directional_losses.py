"""
Loss functions for directional VAE models
"""
import torch
import torch.nn.functional as F
from .losses import nt_xent_loss, triplet_margin_loss_with_hard_negative_mining


def rna2dna_loss(recon_dna, dna, mu, logvar, mu_list=None, labels=None, 
                 beta=1e-3, alpha=0.1, delta=0.1, temperature=0.5, margin=1.0):
    """
    Compute loss for RNA2DNAVAE
    
    Args:
        recon_dna: Reconstructed DNA methylation data
        dna: Original DNA methylation data
        mu: Latent mean
        logvar: Latent log variance
        mu_list: List of individual modality latent means
        labels: Class labels for triplet loss
        beta: Weight for KL divergence term
        alpha: Weight for NT-Xent contrastive loss
        delta: Weight for Triplet Margin Loss
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence, contrastive_loss, triplet_loss)
    """
    # Reconstruction loss (binary cross-entropy for beta values in [0,1])
    recon_loss = F.binary_cross_entropy(recon_dna, dna, reduction='sum')
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Hybrid Contrastive Loss
    cont_loss = torch.tensor(0.0, device=mu.device)
    if mu_list is not None and len(mu_list) >= 2:
        cont_loss = nt_xent_loss(mu_list[0], mu_list[1], temperature=temperature)

    # Triplet Margin Loss
    triplet_loss = torch.tensor(0.0, device=mu.device)
    if labels is not None:
        triplet_loss = triplet_margin_loss_with_hard_negative_mining(mu, labels, margin=margin)

    total_loss = recon_loss + beta * kld + alpha * cont_loss * mu.size(0) + delta * triplet_loss * mu.size(0)
    
    return total_loss, recon_loss.item(), kld.item(), cont_loss.item(), triplet_loss.item()


def dna2rna_loss(recon_rna, rna, mu, logvar, mu_list=None, labels=None,
                 beta=1e-3, alpha=0.1, delta=0.1, temperature=0.5, margin=1.0):
    """
    Compute loss for DNA2RNAVAE
    
    Args:
        recon_rna: Reconstructed RNA expression data
        rna: Original RNA expression data
        mu: Latent mean
        logvar: Latent log variance
        mu_list: List of individual modality latent means
        labels: Class labels for triplet loss
        beta: Weight for KL divergence term
        alpha: Weight for NT-Xent contrastive loss
        delta: Weight for Triplet Margin Loss
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence, contrastive_loss, triplet_loss)
    """
    # Reconstruction loss (MSE for RNA expression)
    recon_loss = F.mse_loss(recon_rna, rna, reduction='sum')
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Hybrid Contrastive Loss
    cont_loss = torch.tensor(0.0, device=mu.device)
    if mu_list is not None and len(mu_list) >= 2:
        cont_loss = nt_xent_loss(mu_list[0], mu_list[1], temperature=temperature)

    # Triplet Margin Loss
    triplet_loss = torch.tensor(0.0, device=mu.device)
    if labels is not None:
        triplet_loss = triplet_margin_loss_with_hard_negative_mining(mu, labels, margin=margin)

    total_loss = recon_loss + beta * kld + alpha * cont_loss * mu.size(0) + delta * triplet_loss * mu.size(0)
    
    return total_loss, recon_loss.item(), kld.item(), cont_loss.item(), triplet_loss.item()

