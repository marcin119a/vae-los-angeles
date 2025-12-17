"""
Loss functions for directional Autoencoder models (non-VAE)
"""
import torch
import torch.nn.functional as F


def rna2dna_ae_loss(recon_dna, dna):
    """
    Compute loss for RNA2DNAAE (reconstruction only, no KL divergence)
    
    Args:
        recon_dna: Reconstructed DNA methylation data
        dna: Original DNA methylation data
        
    Returns:
        Tuple of (total_loss, reconstruction_loss)
    """
    # Reconstruction loss (binary cross-entropy for beta values in [0,1])
    recon_loss = F.binary_cross_entropy(recon_dna, dna, reduction='sum')
    
    return recon_loss, recon_loss.item()


def dna2rna_ae_loss(recon_rna, rna):
    """
    Compute loss for DNA2RNAAE (reconstruction only, no KL divergence)
    
    Args:
        recon_rna: Reconstructed RNA expression data
        rna: Original RNA expression data
        
    Returns:
        Tuple of (total_loss, reconstruction_loss)
    """
    # Reconstruction loss (MSE for RNA expression)
    recon_loss = F.mse_loss(recon_rna, rna, reduction='sum')
    
    return recon_loss, recon_loss.item()

