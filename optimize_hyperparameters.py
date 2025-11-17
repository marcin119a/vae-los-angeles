"""
Hyperparameter optimization for Multi-Modal VAE using Optuna
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import optuna
from datetime import datetime

from src.config import Config
from src.models import MultiModalVAE
from src.data import MultiModalDataset
from src.utils import vae_loss

def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

def load_data():
    """Load processed data"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return merged_df, label_encoder

def compute_class_weights(train_df, n_classes):
    """Compute class weights for balanced loss"""
    class_labels = train_df['primary_site_encoded'].values
    unique_classes = np.unique(class_labels)
    class_weights_present = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=class_labels
    )
    class_weights = np.ones(n_classes, dtype=np.float32)
    class_weights[unique_classes] = class_weights_present
    return torch.FloatTensor(class_weights).to(Config.DEVICE)

def prepare_dataloaders(merged_df):
    """Split data and create dataloaders"""
    train_df, val_df = train_test_split(
        merged_df,
        test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_SEED
    )
    train_dataset = MultiModalDataset(train_df)
    val_dataset = MultiModalDataset(val_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    return train_dataloader, val_dataloader, train_df

def objective(trial):
    """Optuna objective function"""
    # Hyperparameters to tune
    latent_dim = trial.suggest_int('latent_dim', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    beta_start = trial.suggest_float('beta_start', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.5, 5.0)
    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])

    # Load data
    merged_df, label_encoder = load_data()
    n_sites = len(label_encoder.classes_)
    train_dataloader, val_dataloader, train_df = prepare_dataloaders(merged_df)
    class_weights = compute_class_weights(train_df, n_sites)

    # Initialize model
    model = MultiModalVAE(
        Config.INPUT_DIM_A,
        Config.INPUT_DIM_B,
        n_sites,
        latent_dim,
        embed_dim=embed_dim
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Training loop
    best_val_loss = np.inf
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * beta_start
        for tpm, beta_data, site in train_dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
            loss, _, _, _ = vae_loss(
                recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar,
                beta=beta, gamma=gamma, class_weights=class_weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for tpm, beta_data, site in val_dataloader:
                tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
                recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
                loss, _, _, _ = vae_loss(
                    recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar,
                    beta=beta, gamma=gamma, class_weights=class_weights
                )
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_dataloader)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

def main():
    """Main optimization loop"""
    setup_directories()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, timeout=3000)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters
    best_params = trial.params
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\nBest hyperparameters saved to best_hyperparameters.json")

    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    merged_df, label_encoder = load_data()
    n_sites = len(label_encoder.classes_)
    train_dataloader, val_dataloader, train_df = prepare_dataloaders(merged_df)
    class_weights = compute_class_weights(train_df, n_sites)

    model = MultiModalVAE(
        Config.INPUT_DIM_A,
        Config.INPUT_DIM_B,
        n_sites,
        best_params['latent_dim'],
        embed_dim=best_params['embed_dim']
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    best_val_loss = np.inf
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * best_params['beta_start']
        for tpm, beta_data, site in train_dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
            loss, _, _, _ = vae_loss(
                recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar,
                beta=beta, gamma=best_params['gamma'], class_weights=class_weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for tpm, beta_data, site in val_dataloader:
                tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
                recon_a, recon_b, recon_c, mu, logvar = model(a=tpm, b=beta_data, site=site)
                loss, _, _, _ = vae_loss(
                    recon_a, tpm, recon_b, beta_data, recon_c, site, mu, logvar,
                    beta=beta, gamma=best_params['gamma'], class_weights=class_weights
                )
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | Val Loss: {avg_val_loss:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_multivae_optimized.pt')
            torch.save(model.state_dict(), model_path)
            print(f"✓ Best optimized model saved (val_loss: {avg_val_loss:.2f})")

    print("\n" + "="*50)
    print("Hyperparameter optimization complete!")
    print(f"Best model saved to: {os.path.join(Config.CHECKPOINT_DIR, 'best_multivae_optimized.pt')}")
    print("="*50)

if __name__ == "__main__":
    main()
