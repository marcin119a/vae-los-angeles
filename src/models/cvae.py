"""
Conditional VAE (CVAE) models for RNA and DNA autoencoders
"""
import torch
import torch.nn as nn

from .vae import reparameterize


class RNACVAE(nn.Module):
    """
    Conditional VAE for RNA expression data
    
    Encodes: RNA (conditioned on primary_site and RNA)
    Decodes: RNA (conditioned on primary_site and RNA)
    """
    def __init__(self, rna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.rna_dim = rna_dim
        self.latent_dim = latent_dim
        
        # Site embedding
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        # Conditional encoder: takes RNA + (site_embedding + RNA) as condition
        # Input: RNA, Condition: site_embedding + RNA
        condition_dim = embed_dim + rna_dim
        encoder_input_dim = rna_dim + condition_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Conditional decoder: takes latent z + (site_embedding + RNA) as condition
        decoder_input_dim = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rna_dim)
        )

    def forward(self, rna, site):
        """
        Forward pass
        
        Args:
            rna: RNA expression data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_rna, mu, logvar)
        """
        # Get site embedding
        site_emb = self.site_embedding(site)  # (batch_size, embed_dim)
        
        # Create condition: concatenate site_embedding and RNA
        condition = torch.cat([site_emb, rna], dim=1)  # (batch_size, embed_dim + rna_dim)
        
        # Encoder: concatenate RNA with condition
        encoder_input = torch.cat([rna, condition], dim=1)  # (batch_size, rna_dim + condition_dim)
        h = self.encoder(encoder_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization
        z = reparameterize(mu, logvar)
        
        # Decoder: concatenate latent z with condition
        decoder_input = torch.cat([z, condition], dim=1)  # (batch_size, latent_dim + condition_dim)
        recon_rna = self.decoder(decoder_input)
        
        return recon_rna, mu, logvar


class DNACVAE(nn.Module):
    """
    Conditional VAE for DNA methylation data
    
    Encodes: DNA (conditioned on primary_site and DNA)
    Decodes: DNA (conditioned on primary_site and DNA)
    """
    def __init__(self, dna_dim, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        self.dna_dim = dna_dim
        self.latent_dim = latent_dim
        
        # Site embedding
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        # Conditional encoder: takes DNA + (site_embedding + DNA) as condition
        # Input: DNA, Condition: site_embedding + DNA
        condition_dim = embed_dim + dna_dim
        encoder_input_dim = dna_dim + condition_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Conditional decoder: takes latent z + (site_embedding + DNA) as condition
        decoder_input_dim = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, dna_dim),
            nn.Sigmoid()  # Beta-values in range 0-1
        )

    def forward(self, dna, site):
        """
        Forward pass
        
        Args:
            dna: DNA methylation data
            site: Primary site labels
            
        Returns:
            Tuple of (reconstructed_dna, mu, logvar)
        """
        # Flatten DNA if needed
        dna_flat = dna.view(dna.size(0), -1)
        
        # Get site embedding
        site_emb = self.site_embedding(site)  # (batch_size, embed_dim)
        
        # Create condition: concatenate site_embedding and DNA
        condition = torch.cat([site_emb, dna_flat], dim=1)  # (batch_size, embed_dim + dna_dim)
        
        # Encoder: concatenate DNA with condition
        encoder_input = torch.cat([dna_flat, condition], dim=1)  # (batch_size, dna_dim + condition_dim)
        h = self.encoder(encoder_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization
        z = reparameterize(mu, logvar)
        
        # Decoder: concatenate latent z with condition
        decoder_input = torch.cat([z, condition], dim=1)  # (batch_size, latent_dim + condition_dim)
        recon_dna = self.decoder(decoder_input)
        
        return recon_dna, mu, logvar
