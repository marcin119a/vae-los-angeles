"""
Training script for Conditional Multi-Modal VAE (CVAE)
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

from src.models.cvae import ConditionalMultiModalVAE
from src.data.dataset import MultiModalDataset
from src.utils.losses import vae_loss
from src.config import Config


def train_epoch(model, dataloader, optimizer, device, beta, gamma):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_class = 0
    train_kld = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Get data (batch is tuple: (rna, dna, site))
        rna = batch[0].to(device)
        dna = batch[1].to(device)
        site = batch[2].to(device)
        
        # Forward pass (site is REQUIRED for CVAE)
            
        recon_rna, recon_dna, recon_site, mu, logvar = model(rna, dna, site)
        
        # Compute loss
        loss, recon, class_loss, kld = vae_loss(
            recon_rna, rna,
            recon_dna, dna,
            recon_site, site,
            mu, logvar,
            beta=beta,
            gamma=gamma
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        train_loss += loss.item()
        train_recon += recon
        train_class += class_loss
        train_kld += kld
    
    n_batches = len(dataloader)
    return (train_loss / n_batches, 
            train_recon / n_batches,
            train_class / n_batches,
            train_kld / n_batches)


def validate(model, dataloader, device, beta, gamma):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_recon = 0
    val_class = 0
    val_kld = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data (batch is tuple: (rna, dna, site))
            rna = batch[0].to(device)
            dna = batch[1].to(device)
            site = batch[2].to(device)
            
            recon_rna, recon_dna, recon_site, mu, logvar = model(rna, dna, site)
            
            loss, recon, class_loss, kld = vae_loss(
                recon_rna, rna,
                recon_dna, dna,
                recon_site, site,
                mu, logvar,
                beta=beta,
                gamma=gamma
            )
            
            val_loss += loss.item()
            val_recon += recon
            val_class += class_loss
            val_kld += kld
    
    n_batches = len(dataloader)
    return (val_loss / n_batches,
            val_recon / n_batches,
            val_class / n_batches,
            val_kld / n_batches)


def main():
    # Configuration
    device = Config.DEVICE
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Data shape: {merged_df.shape}")
    print(f"Number of primary sites: {len(label_encoder.classes_)}")
    
    # Split data
    print('Splitting data into train/validation sets...')
    train_df, val_df = train_test_split(
        merged_df,
        test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_SEED
    )
    
    print(f'Train set size: {len(train_df)}')
    print(f'Validation set size: {len(val_df)}')
    
    # Create datasets
    train_dataset = MultiModalDataset(train_df)
    val_dataset = MultiModalDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize CVAE model
    print('Initializing CVAE model...')
    # Get dimensions from dataset
    sample_batch = next(iter(train_loader))
    input_dim_rna = sample_batch[0].shape[1]  # tpm (RNA)
    input_dim_dna = sample_batch[1].shape[1]   # beta (DNA)
    n_sites = len(label_encoder.classes_)
    
    model = ConditionalMultiModalVAE(
        input_dim_a=input_dim_rna,
        input_dim_b=input_dim_dna,
        n_sites=n_sites,
        latent_dim=Config.LATENT_DIM,
        embed_dim=32
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_recon, train_class, train_kld = train_epoch(
            model, train_loader, optimizer, device, Config.BETA_START, Config.GAMMA
        )
        
        # Validate
        val_loss, val_recon, val_class, val_kld = validate(
            model, val_loader, device, Config.BETA_START, Config.GAMMA
        )
        
        # Print progress in compact format
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.2f} | "
              f"Val Loss: {val_loss:.2f} | "
              f"Recon: {val_recon:.2f} | "
              f"Class: {val_class:.2f} | "
              f"KLD: {val_kld:.2f}")
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = 'checkpoints/best_cvae.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'latent_dim': Config.LATENT_DIM,
                'beta': Config.BETA_START,
                'gamma': Config.GAMMA
            }, save_path)
            print(f"✓ Best model saved (val_loss: {val_loss:.2f})")
        else:
            patience_counter += 1
            
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
    
    print("\n" + "="*60)
    print("CVAE Training Complete!")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Model saved to: checkpoints/best_cvae.pt")
    print("="*60)


if __name__ == '__main__':
    main()

