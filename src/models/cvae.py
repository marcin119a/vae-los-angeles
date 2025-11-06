"""
Conditional Multi-Modal VAE model (CVAE)
"""
import torch
import torch.nn as nn

from .vae import reparameterize


class ConditionalEncoderA(nn.Module):
    """Conditional Encoder for RNA expression data (modality A)"""
    def __init__(self, input_dim, latent_dim, n_sites, embed_dim=32):
        super().__init__()
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, condition):
        """
        Args:
            x: RNA expression data
            condition: site labels
        """
        c = self.site_embedding(condition)
        x_c = torch.cat([x, c], dim=1)
        h = self.fc(x_c)
        return self.fc_mu(h), self.fc_logvar(h)


class ConditionalEncoderB(nn.Module):
    """Conditional Encoder for DNA methylation data (modality B)"""
    def __init__(self, input_dim, latent_dim, n_sites, embed_dim=32):
        super().__init__()
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 512),
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

    def forward(self, x, condition):
        """
        Args:
            x: DNA methylation data
            condition: site labels
        """
        x = x.view(x.size(0), -1)
        c = self.site_embedding(condition)
        x_c = torch.cat([x, c], dim=1)
        h = self.fc(x_c)
        return self.fc_mu(h), self.fc_logvar(h)


class ConditionalDecoderA(nn.Module):
    """Conditional Decoder for RNA expression data (modality A)"""
    def __init__(self, latent_dim, output_dim, n_sites, embed_dim=32):
        super().__init__()
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z, condition):
        """
        Args:
            z: latent vector
            condition: site labels
        """
        c = self.site_embedding(condition)
        z_c = torch.cat([z, c], dim=1)
        return self.fc(z_c)


class ConditionalDecoderB(nn.Module):
    """Conditional Decoder for DNA methylation data (modality B)"""
    def __init__(self, latent_dim, output_dim, n_sites, embed_dim=32):
        super().__init__()
        self.site_embedding = nn.Embedding(n_sites, embed_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Beta-values in range 0-1
        )

    def forward(self, z, condition):
        """
        Args:
            z: latent vector
            condition: site labels
        """
        c = self.site_embedding(condition)
        z_c = torch.cat([z, c], dim=1)
        return self.fc(z_c)


class ConditionalDecoderC(nn.Module):
    """Decoder for primary site classification (modality C)"""
    def __init__(self, latent_dim, n_sites):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_sites)
        )

    def forward(self, z):
        """No conditioning needed for classification output"""
        return self.fc(z)


class ConditionalMultiModalVAE(nn.Module):
    """
    Conditional Multi-Modal Variational Autoencoder (CVAE)
    
    Conditions the generation on primary site labels.
    Supports three modalities:
    - A: RNA expression data
    - B: DNA methylation data
    - C: Primary site labels (used as condition)
    """
    def __init__(self, input_dim_a, input_dim_b, n_sites, latent_dim, embed_dim=32):
        super().__init__()
        
        # Conditional encoders
        self.encoder_a = ConditionalEncoderA(input_dim_a, latent_dim, n_sites, embed_dim)
        self.encoder_b = ConditionalEncoderB(input_dim_b, latent_dim, n_sites, embed_dim)
        
        # Non-conditional encoder for site (for when we need to infer from site only)
        from .encoders import EncoderC
        self.encoder_c = EncoderC(n_sites, latent_dim, embed_dim)
        
        # Conditional decoders
        self.decoder_a = ConditionalDecoderA(latent_dim, input_dim_a, n_sites, embed_dim)
        self.decoder_b = ConditionalDecoderB(latent_dim, input_dim_b, n_sites, embed_dim)
        self.decoder_c = ConditionalDecoderC(latent_dim, n_sites)

    def forward(self, a=None, b=None, site=None):
        """
        Forward pass through the CVAE
        
        Args:
            a: RNA expression data (optional)
            b: DNA methylation data (optional)
            site: Primary site labels (REQUIRED for CVAE)
            
        Returns:
            Tuple of (out_a, out_b, out_c, mu, logvar)
        """
        if site is None:
            raise ValueError("CVAE requires site labels as conditioning information")
        
        mu_list, logvar_list = [], []

        # Encode with conditioning
        if a is not None:
            mu_a, logvar_a = self.encoder_a(a, site)
            mu_list.append(mu_a)
            logvar_list.append(logvar_a)
            
        if b is not None:
            mu_b, logvar_b = self.encoder_b(b, site)
            mu_list.append(mu_b)
            logvar_list.append(logvar_b)

        # If only site is provided, use non-conditional encoder
        if len(mu_list) == 0:
            mu, logvar = self.encoder_c(site)
        elif len(mu_list) == 1:
            mu, logvar = mu_list[0], logvar_list[0]
        else:
            # Aggregate latent representations
            mu = torch.stack(mu_list).mean(0)
            logvar = torch.stack(logvar_list).mean(0)

        # Reparameterization trick
        z = reparameterize(mu, logvar)

        # Decode with conditioning
        out_a = self.decoder_a(z, site)
        out_b = self.decoder_b(z, site)
        out_c = self.decoder_c(z)  # No conditioning for classification

        return out_a, out_b, out_c, mu, logvar

    def generate(self, site, n_samples=1, device='cpu'):
        """
        Generate samples conditioned on site labels
        
        Args:
            site: site labels (tensor of shape [n_samples])
            n_samples: number of samples to generate
            device: device to use
            
        Returns:
            Tuple of (generated_a, generated_b, predicted_site)
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.decoder_a.fc[0].in_features - 
                          self.decoder_a.site_embedding.embedding_dim).to(device)
            
            # Decode with conditioning
            gen_a = self.decoder_a(z, site)
            gen_b = self.decoder_b(z, site)
            gen_c = self.decoder_c(z)
            
        return gen_a, gen_b, gen_c

