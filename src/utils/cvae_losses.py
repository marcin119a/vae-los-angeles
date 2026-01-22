"""
Loss functions for Conditional VAE (CVAE) models
"""
import torch
import torch.nn.functional as F


def rna_cvae_loss(recon_rna, rna, mu, logvar, beta=1e-3):
    """
    Compute loss for RNACVAE
    
    Args:
        recon_rna: Reconstructed RNA expression data
        rna: Original RNA expression data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (MSE for RNA expression)
    recon_loss = F.mse_loss(recon_rna, rna, reduction='sum')
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss.item(), kld.item()


def dna_cvae_loss(recon_dna, dna, mu, logvar, beta=1e-3):
    """
    Compute loss for DNACVAE
    
    Args:
        recon_dna: Reconstructed DNA methylation data
        dna: Original DNA methylation data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Flatten DNA if needed
    dna_flat = dna.view(dna.size(0), -1)
    
    # Reconstruction loss (binary cross-entropy for beta values in [0,1])
    recon_loss = F.binary_cross_entropy(recon_dna, dna_flat, reduction='sum')
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss.item(), kld.item()
