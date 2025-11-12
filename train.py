"""
Training script for Multi-Modal VAE
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import MultiModalVAE
from src.data import MultiModalDataset
from src.utils import vae_loss


def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def load_data():
    """Load processed data"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Data shape: {merged_df.shape}")
    print(f"Number of primary sites: {len(label_encoder.classes_)}")
    
    return merged_df, label_encoder


def compute_class_weights(train_df, n_classes):
    """Compute class weights for balanced loss"""
    print("\nComputing class weights for balanced classification loss...")
    
    # Get class labels from training data
    class_labels = train_df['primary_site_encoded'].values
    
    # Get unique classes present in training data
    unique_classes = np.unique(class_labels)
    
    # Compute class weights only for classes present in training data
    class_weights_present = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_labels
    )
    
    # Create full weight tensor with 1.0 for missing classes (won't be used in training)
    class_weights = np.ones(n_classes, dtype=np.float32)
    class_weights[unique_classes] = class_weights_present
    
    # Convert to tensor
    class_weights_tensor = torch.FloatTensor(class_weights).to(Config.DEVICE)
    
    # Print class distribution and weights
    unique, counts = np.unique(class_labels, return_counts=True)
    print(f"Class distribution in training set:")
    print(f"  Total classes in training set: {len(unique)} out of {n_classes}")
    for cls, count in zip(unique[:5], counts[:5]):
        print(f"  Class {cls}: {count} samples, weight: {class_weights[cls]:.4f}")
    if len(unique) > 5:
        print(f"  ... and {len(unique) - 5} more classes")
    
    return class_weights_tensor


def prepare_dataloaders(merged_df):
    """Split data and create dataloaders"""
    print("\nSplitting data into train/validation sets...")
    train_df, val_df = train_test_split(
        merged_df, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Create datasets
    train_dataset = MultiModalDataset(train_df)
    val_dataset = MultiModalDataset(val_df)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, train_df


def train_epoch(model, dataloader, optimizer, epoch, class_weights):
    """Train for one epoch"""
    model.train()
    running_train_loss = 0.0
    recon_component = 0.0
    kl_component = 0.0
    class_component = 0.0
    
    # Beta warmup: gradually increase KL weight
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START
    
    for tpm, beta_data, site in dataloader:
        tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
        
        # Forward pass
        recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
        
        # Compute loss with class weights
        loss, recon_loss, class_loss, kld_loss = vae_loss(
            recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar,
            beta=beta, gamma=Config.GAMMA, class_weights=class_weights
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_train_loss += loss.item()
        recon_component += recon_loss
        kl_component += kld_loss
        class_component += class_loss
    
    avg_train_loss = running_train_loss / len(dataloader)
    
    return avg_train_loss, beta


def validate(model, dataloader, epoch, class_weights):
    """Validate the model"""
    model.eval()
    running_val_loss = 0.0
    
    beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            
            # Forward pass
            recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
            
            # Compute loss with class weights
            loss, _, _, _ = vae_loss(
                recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar, 
                beta=beta, class_weights=class_weights
            )
            
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(dataloader)
    
    return avg_val_loss


def plot_losses(train_losses, val_losses, run_id):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss for MultiModalVAE (β-VAE + Warm-up)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    filename = f'plots/training_losses_{run_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {filename}")


def main():
    """Main training loop"""
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting training run: {run_id}")
    
    # Setup
    setup_directories()
    
    # Load data
    merged_df, label_encoder = load_data()
    n_sites = len(label_encoder.classes_)
    
    # Prepare dataloaders
    train_dataloader, val_dataloader, train_df = prepare_dataloaders(merged_df)
    
    # Compute class weights for balanced classification loss
    class_weights = compute_class_weights(train_df, n_sites)
    
    # Initialize model
    print(f"\nInitializing model on {Config.DEVICE}...")
    model = MultiModalVAE(
        Config.INPUT_DIM_A, 
        Config.INPUT_DIM_B, 
        n_sites, 
        Config.LATENT_DIM
    ).to(Config.DEVICE)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.LR_SCHEDULER_FACTOR, 
        patience=Config.LR_SCHEDULER_PATIENCE
    )
    
    # Training loop
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {Config.PATIENCE}")
    
    best_val_loss = np.inf
    trigger = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        avg_train_loss, beta = train_epoch(model, train_dataloader, optimizer, epoch, class_weights)
        train_losses.append(avg_train_loss)
        
        # Validate
        avg_val_loss = validate(model, val_dataloader, epoch, class_weights)
        val_losses.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.2f} | "
              f"Val Loss: {avg_val_loss:.2f} | "
              f"β={beta:.5f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger = 0
            
            # Save best model
            model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_multivae_{run_id}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best model saved (val_loss: {avg_val_loss:.2f})")
        else:
            trigger += 1
            if trigger >= Config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}!")
                break
    
    # Plot losses
    print("\nGenerating loss plots...")
    plot_losses(train_losses, val_losses, run_id)
    
    # Save final model path for evaluation
    with open('latest_run_id.txt', 'w') as f:
        f.write(run_id)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Best model saved to: {os.path.join(Config.CHECKPOINT_DIR, f'best_multivae_{run_id}.pt')}")
    print("="*50)


if __name__ == "__main__":
    main()

