"""
Loss functions for Multi-Modal VAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy loss for cross-modal alignment.
    
    Args:
        z_i, z_j: Latent representations from two different modalities (N x D)
        temperature: Temperature scaling parameter
    """
    N = z_i.size(0)
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    
    # Mask to remove self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim = sim.masked_fill(mask, -float('inf'))
    
    # Labels: the positive pair for i is i+N, and for i+N is i
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(z.device)
    
    loss = F.cross_entropy(sim, labels)
    return loss


def triplet_margin_loss_with_hard_negative_mining(anchor, labels, margin=1.0):
    """
    Triplet Margin Loss with Hard Negative Mining for intra-class separation.
    
    Args:
        anchor: Latent representations (N x D)
        labels: Class labels (N)
        margin: Margin for triplet loss
    """
    if anchor.size(0) <= 1:
        return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(anchor, anchor, p=2)
    
    N = anchor.size(0)
    loss = 0
    count = 0
    
    for i in range(N):
        # Same class indices (excluding self)
        pos_indices = (labels == labels[i]).nonzero().flatten()
        pos_indices = pos_indices[pos_indices != i]
        
        # Different class indices
        neg_indices = (labels != labels[i]).nonzero().flatten()
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Hardest positive (maximum distance among positives)
            hardest_pos = dist_matrix[i, pos_indices].max()
            
            # Hardest negative (minimum distance among negatives)
            hardest_neg = dist_matrix[i, neg_indices].min()
            
            loss += F.relu(hardest_pos - hardest_neg + margin)
            count += 1
            
    if count > 0:
        return loss / count
    else:
        # If no valid triplets in batch, return 0 with grad
        return torch.tensor(0.0, device=anchor.device, requires_grad=True)


def vae_loss(recon_a, a, recon_b, b, recon_c, site, mu, logvar, 
             mu_list=None, labels=None,
             beta=1e-3, gamma=1.0, alpha=0.1, delta=0.1, 
             class_weights=None, temperature=0.5, margin=1.0):
    """
    Compute the Hybrid VAE loss function
    
    Args:
        recon_a: Reconstructed RNA expression data
        a: Original RNA expression data
        recon_b: Reconstructed DNA methylation data
        b: Original DNA methylation data
        recon_c: Reconstructed primary site predictions
        site: Original primary site labels
        mu: Aggregated latent mean
        logvar: Aggregated latent log variance
        mu_list: List of individual modality latent means for cross-modal alignment
        labels: Class labels for triplet loss (usually same as site)
        beta: Weight for KL divergence term
        gamma: Weight for classification loss
        alpha: Weight for NT-Xent contrastive loss
        delta: Weight for Triplet Margin Loss
        class_weights: Tensor with class weights for balanced classification loss
        temperature: Temperature for NT-Xent
        margin: Margin for Triplet Loss
        
    Returns:
        Tuple of (total_loss, recon_loss, class_loss, kld, contrastive_loss, triplet_loss)
    """
    # 1. Reconstruction loss
    recon = 0
    if recon_a is not None and a is not None:
        recon += F.mse_loss(recon_a, a, reduction='sum')
    if recon_b is not None and b is not None:
        # Use binary cross-entropy for DNA methylation data (beta values in [0,1])
        recon += F.binary_cross_entropy(recon_b, b, reduction='sum')

    # 2. Classification loss with class balancing
    class_loss = torch.tensor(0.0, device=mu.device)
    if recon_c is not None and site is not None:
        class_loss = F.cross_entropy(recon_c, site, weight=class_weights, reduction='sum')

    # 3. KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 4. Hybrid Contrastive Loss (Cross-modal alignment)
    cont_loss = torch.tensor(0.0, device=mu.device)
    if mu_list is not None and len(mu_list) >= 2:
        # Compute NT-Xent between pairs of modalities
        pairs_count = 0
        for i in range(len(mu_list)):
            for j in range(i + 1, len(mu_list)):
                cont_loss += nt_xent_loss(mu_list[i], mu_list[j], temperature=temperature)
                pairs_count += 1
        if pairs_count > 0:
            cont_loss /= pairs_count

    # 5. Triplet Margin Loss (Intra-class separation)
    triplet_loss = torch.tensor(0.0, device=mu.device)
    if labels is not None:
        triplet_loss = triplet_margin_loss_with_hard_negative_mining(mu, labels, margin=margin)

    # Combined Loss
    # Scale reconstruction, classification and KLD by batch size for balance if needed, 
    # but here we follow the original 'sum' reduction.
    total_loss = recon + gamma * class_loss + beta * kld + alpha * cont_loss * mu.size(0) + delta * triplet_loss * mu.size(0)
    
    return total_loss, recon.item(), class_loss.item(), kld.item(), cont_loss.item(), triplet_loss.item()

