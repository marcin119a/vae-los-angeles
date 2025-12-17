"""
Directional Autoencoder models for imputation (non-VAE version)
"""
import torch
import torch.nn as nn

from .decoders import DecoderA, DecoderB


class RNA2DNAAE(nn.Module):
    """
    Autoencoder that predicts DNA methylation from RNA expression and primary site
    
    Encodes: RNA + Primary Site -> Latent
    Decodes: Latent -> DNA methylation
    """
    def __init__(self, rna_dim, dna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        # Encoder for RNA (simple encoder, no mu/logvar)
        self.encoder_rna = nn.Sequential(
            nn.Linear(rna_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim)
        )
        
        # Encoder for site (embedding + projection)
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        self.site_projection = nn.Linear(embed_dim, latent_dim)
        
        # Decoder for DNA
        self.decoder_dna = DecoderB(latent_dim, dna_dim)

    def forward(self, rna=None, site=None):
        """
        Forward pass
        
        Args:
            rna: RNA expression data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_dna, latent) - no mu/logvar for AE
        """
        latent_list = []

        if rna is not None:
            latent_rna = self.encoder_rna(rna)
            latent_list.append(latent_rna)
            
        if site is not None:
            # Handle embedding separately
            site_emb = self.site_embedding(site)  # Embedding
            latent_site = self.site_projection(site_emb)  # Projection
            latent_list.append(latent_site)

        # Aggregate latent representations (mean)
        if len(latent_list) == 0:
            return None, None
        elif len(latent_list) == 1:
            latent = latent_list[0]
        else:
            latent = torch.stack(latent_list).mean(0)

        recon_dna = self.decoder_dna(latent)

        return recon_dna, latent


class DNA2RNAAE(nn.Module):
    """
    Autoencoder that predicts RNA expression from DNA methylation and primary site
    
    Encodes: DNA + Primary Site -> Latent
    Decodes: Latent -> RNA expression
    """
    def __init__(self, rna_dim, dna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        # Encoder for DNA (simple encoder, no mu/logvar)
        self.encoder_dna = nn.Sequential(
            nn.Linear(dna_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        # Encoder for site (embedding + projection)
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        self.site_projection = nn.Linear(embed_dim, latent_dim)
        
        # Decoder for RNA
        self.decoder_rna = DecoderA(latent_dim, rna_dim)

    def forward(self, dna=None, site=None):
        """
        Forward pass
        
        Args:
            dna: DNA methylation data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_rna, latent) - no mu/logvar for AE
        """
        latent_list = []

        if dna is not None:
            dna_flat = dna.view(dna.size(0), -1)
            latent_dna = self.encoder_dna(dna_flat)
            latent_list.append(latent_dna)
            
        if site is not None:
            # Handle embedding separately
            site_emb = self.site_embedding(site)  # Embedding
            latent_site = self.site_projection(site_emb)  # Projection
            latent_list.append(latent_site)

        # Aggregate latent representations (mean)
        if len(latent_list) == 0:
            return None, None
        elif len(latent_list) == 1:
            latent = latent_list[0]
        else:
            latent = torch.stack(latent_list).mean(0)

        recon_rna = self.decoder_rna(latent)

        return recon_rna, latent

