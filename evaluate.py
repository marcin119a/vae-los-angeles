"""
Evaluation script for Multi-Modal VAE
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import MultiModalVAE
from src.data import MultiModalDataset


def get_run_id():
    """Get the run ID from latest training or use default"""
    if os.path.exists('latest_run_id.txt'):
        with open('latest_run_id.txt', 'r') as f:
            return f.read().strip()
    else:
        # Use default model name if no run_id file exists
        return None


def load_model_and_data():
    """Load trained model and validation data"""
    run_id = get_run_id()
    
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    n_sites = len(label_encoder.classes_)
    
    # Split data
    _, val_df = train_test_split(
        merged_df, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    # Create validation dataloader
    val_dataset = MultiModalDataset(val_df)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    # Load model
    if run_id:
        model_filename = f'best_multivae_{run_id}.pt'
        print(f"Loading model from run: {run_id}")
    else:
        model_filename = Config.BEST_MODEL_NAME
        print(f"Loading default model: {model_filename}")
    
    model = MultiModalVAE(
        Config.INPUT_DIM_A, 
        Config.INPUT_DIM_B, 
        n_sites, 
        Config.LATENT_DIM
    ).to(Config.DEVICE)
    
    model_path = os.path.join(Config.CHECKPOINT_DIR, model_filename)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, val_dataloader, run_id


def reconstruct_all_samples(model, dataloader):
    """Reconstruct all samples in the validation set"""
    print("\nReconstructing all validation samples...")
    
    all_orig_rna, all_recon_rna = [], []
    all_orig_beta, all_recon_beta = [], []
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            
            # Cross-modal reconstructions
            recon_a_from_b, _, _, _, _ = model(a=None, b=beta_data, site=site)
            _, recon_b_from_a, _, _, _ = model(a=tpm, b=None, site=site)
            
            all_orig_rna.append(tpm.cpu().numpy())
            all_recon_rna.append(recon_a_from_b.cpu().numpy())
            all_orig_beta.append(beta_data.cpu().numpy())
            all_recon_beta.append(recon_b_from_a.cpu().numpy())
    
    # Concatenate all batches
    orig_rna = np.concatenate(all_orig_rna, axis=0)
    recon_rna = np.concatenate(all_recon_rna, axis=0)
    orig_beta = np.concatenate(all_orig_beta, axis=0)
    recon_beta = np.concatenate(all_recon_beta, axis=0)
    
    return orig_rna, recon_rna, orig_beta, recon_beta


def compute_metrics(orig, recon, modality_name):
    """Compute reconstruction metrics"""
    print(f"\nComputing metrics for {modality_name}...")
    
    # MSE and MAE
    mse = mean_squared_error(orig.flatten(), recon.flatten())
    mae = mean_absolute_error(orig.flatten(), recon.flatten())
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        torch.tensor(orig), 
        torch.tensor(recon), 
        dim=1
    ).mean().item()
    
    # Pearson correlation (per sample)
    pearson_all = []
    for i in range(len(orig)):
        try:
            r, _ = pearsonr(orig[i], recon[i])
            if not np.isnan(r):
                pearson_all.append(r)
        except:
            pass
    
    pearson_mean = np.mean(pearson_all) if pearson_all else 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cos_sim,
        'pearson_mean': pearson_mean,
        'pearson_all': pearson_all
    }


def plot_reconstruction_examples(orig_rna, recon_rna, orig_beta, recon_beta, run_id, n_examples=3):
    """Plot example reconstructions"""
    print(f"\nGenerating reconstruction plots ({n_examples} examples)...")
    
    indices = np.random.choice(len(orig_rna), n_examples, replace=False)
    
    run_suffix = f"_{run_id}" if run_id else ""
    
    for idx_num, idx in enumerate(indices):
        plt.figure(figsize=(12, 4))
        
        # RNA reconstruction
        plt.subplot(1, 2, 1)
        plt.plot(orig_rna[idx], label="Original RNA", alpha=0.7)
        plt.plot(recon_rna[idx], label="Reconstructed from DNA", alpha=0.7)
        plt.title("RNA reconstructed from DNA methylation")
        plt.xlabel("Gene index")
        plt.ylabel("Expression (normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # DNA methylation reconstruction
        plt.subplot(1, 2, 2)
        plt.plot(orig_beta[idx], label="Original DNA methylation", alpha=0.7)
        plt.plot(recon_beta[idx], label="Reconstructed from RNA", alpha=0.7)
        plt.title("DNA methylation reconstructed from RNA")
        plt.xlabel("Probe index")
        plt.ylabel("Beta value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/reconstruction_example_{idx_num+1}{run_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_examples} reconstruction examples to plots/")


def plot_correlation_distributions(metrics_rna, metrics_beta, run_id):
    """Plot Pearson correlation distributions"""
    print("\nGenerating correlation distribution plots...")
    
    run_suffix = f"_{run_id}" if run_id else ""
    
    plt.figure(figsize=(12, 5))
    
    # RNA correlation distribution
    plt.subplot(1, 2, 1)
    plt.hist(
        metrics_rna['pearson_all'], 
        bins=30, 
        color='skyblue', 
        edgecolor='black', 
        alpha=0.7
    )
    plt.axvline(
        metrics_rna['pearson_mean'], 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {metrics_rna["pearson_mean"]:.3f}'
    )
    plt.title("Pearson r distribution (RNA ← DNA)")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DNA correlation distribution
    plt.subplot(1, 2, 2)
    plt.hist(
        metrics_beta['pearson_all'], 
        bins=30, 
        color='salmon', 
        edgecolor='black', 
        alpha=0.7
    )
    plt.axvline(
        metrics_beta['pearson_mean'], 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {metrics_beta["pearson_mean"]:.3f}'
    )
    plt.title("Pearson r distribution (DNA ← RNA)")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'plots/correlation_distributions{run_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distribution plots to {filename}")


def print_results(metrics_rna, metrics_beta):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n===== RNA reconstructed from DNA methylation =====")
    print(f"  MSE:                {metrics_rna['mse']:.4f}")
    print(f"  MAE:                {metrics_rna['mae']:.4f}")
    print(f"  Cosine similarity:  {metrics_rna['cosine_similarity']:.4f}")
    print(f"  Mean Pearson r:     {metrics_rna['pearson_mean']:.4f}")
    
    print("\n===== DNA methylation reconstructed from RNA =====")
    print(f"  MSE:                {metrics_beta['mse']:.4f}")
    print(f"  MAE:                {metrics_beta['mae']:.4f}")
    print(f"  Cosine similarity:  {metrics_beta['cosine_similarity']:.4f}")
    print(f"  Mean Pearson r:     {metrics_beta['pearson_mean']:.4f}")
    
    print("\n" + "="*60)


def save_results(metrics_rna, metrics_beta, run_id):
    """Save evaluation results to file"""
    run_suffix = f"_{run_id}" if run_id else ""
    
    results = {
        'run_id': run_id if run_id else 'default',
        'rna_from_dna': {
            'mse': float(metrics_rna['mse']),
            'mae': float(metrics_rna['mae']),
            'cosine_similarity': float(metrics_rna['cosine_similarity']),
            'pearson_mean': float(metrics_rna['pearson_mean']),
        },
        'dna_from_rna': {
            'mse': float(metrics_beta['mse']),
            'mae': float(metrics_beta['mae']),
            'cosine_similarity': float(metrics_beta['cosine_similarity']),
            'pearson_mean': float(metrics_beta['pearson_mean']),
        }
    }
    
    import json
    filename = f'plots/evaluation_results{run_suffix}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    """Main evaluation pipeline"""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load model and data
    model, val_dataloader, run_id = load_model_and_data()
    
    # Reconstruct all samples
    orig_rna, recon_rna, orig_beta, recon_beta = reconstruct_all_samples(
        model, val_dataloader
    )
    
    print(f"Total validation samples: {len(orig_rna)}")
    
    # Compute metrics
    metrics_rna = compute_metrics(orig_rna, recon_rna, "RNA from DNA")
    metrics_beta = compute_metrics(orig_beta, recon_beta, "DNA from RNA")
    
    # Generate plots
    plot_reconstruction_examples(orig_rna, recon_rna, orig_beta, recon_beta, run_id, n_examples=3)
    plot_correlation_distributions(metrics_rna, metrics_beta, run_id)
    
    # Print and save results
    print_results(metrics_rna, metrics_beta)
    save_results(metrics_rna, metrics_beta, run_id)
    
    print("\nEvaluation complete! All results saved to plots/")


if __name__ == "__main__":
    main()

