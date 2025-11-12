"""
Loss functions for Multi-Modal VAE
"""
import torch
import torch.nn.functional as F


def vae_loss(recon_a, a, recon_b, b, recon_c, site, mu, logvar, beta=1e-3, gamma=1.0, class_weights=None):
    """
    Compute the VAE loss function
    
    Args:
        recon_a: Reconstructed RNA expression data
        a: Original RNA expression data
        recon_b: Reconstructed DNA methylation data
        b: Original DNA methylation data
        recon_c: Reconstructed primary site predictions
        site: Original primary site labels
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
        gamma: Weight for classification loss
        class_weights: Tensor with class weights for balanced classification loss
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, classification_loss, kl_divergence)
    """
    # Reconstruction loss
    recon = 0
    if recon_a is not None and a is not None:
        recon += F.mse_loss(recon_a, a, reduction='sum')
    if recon_b is not None and b is not None:
        # Use binary cross-entropy for DNA methylation data (beta values in [0,1])
        recon += F.binary_cross_entropy(recon_b, b, reduction='sum')

    # Classification loss with class balancing
    class_loss = 0
    if recon_c is not None and site is not None:
        class_loss = F.cross_entropy(recon_c, site, weight=class_weights, reduction='sum')

    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon + gamma * class_loss + beta * kld
    
    return total_loss, recon.item(), class_loss.item(), kld.item()

