"""
Evaluation script for Conditional VAE (CVAE) models
Evaluates reconstruction quality for RNA and DNA CVAE
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import RNACVAE, DNACVAE
from src.data import MultiModalDataset


def get_run_id(model_type):
    """Get the run ID from latest training or use default"""
    filename = f'latest_{model_type}_cvae_run_id.txt'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read().strip()
    else:
        return None


def load_model_and_data(model_type):
    """Load trained CVAE model and validation data"""
    run_id = get_run_id(model_type)
    
    print(f"Loading processed data for {model_type} CVAE...")
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
    if model_type == "rna":
        if run_id:
            model_filename = f'best_rna_cvae_{run_id}.pt'
            print(f"Loading RNACVAE from run: {run_id}")
        else:
            print("Warning: No RNACVAE run ID found. Please train the model first.")
            return None, None, None
        
        model = RNACVAE(
            Config.INPUT_DIM_A,  # RNA dimension
            n_sites, 
            Config.LATENT_DIM
        ).to(Config.DEVICE)
        
    elif model_type == "dna":
        if run_id:
            model_filename = f'best_dna_cvae_{run_id}.pt'
            print(f"Loading DNACVAE from run: {run_id}")
        else:
            print("Warning: No DNACVAE run ID found. Please train the model first.")
            return None, None, None
        
        model = DNACVAE(
            Config.INPUT_DIM_B,  # DNA dimension
            n_sites, 
            Config.LATENT_DIM
        ).to(Config.DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model_path = os.path.join(Config.CHECKPOINT_DIR, model_filename)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return None, None, None
    
    return model, val_dataloader, run_id


def reconstruct_all_samples(model, dataloader, model_type):
    """Reconstruct all samples in the validation set"""
    print(f"\nReconstructing all validation samples using {model_type.upper()} CVAE...")
    
    all_orig, all_recon = [], []
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm, beta_data, site = tpm.to(Config.DEVICE), beta_data.to(Config.DEVICE), site.to(Config.DEVICE)
            
            if model_type == "rna":
                recon, _, _ = model(rna=tpm, site=site)
                all_orig.append(tpm.cpu().numpy())
                all_recon.append(recon.cpu().numpy())
            else:  # dna
                recon, _, _ = model(dna=beta_data, site=site)
                all_orig.append(beta_data.cpu().numpy())
                all_recon.append(recon.cpu().numpy())
    
    # Concatenate all batches
    orig = np.concatenate(all_orig, axis=0)
    recon = np.concatenate(all_recon, axis=0)
    
    return orig, recon


def compute_metrics(orig, recon, modality_name):
    """Compute reconstruction metrics"""
    print(f"\nComputing metrics for {modality_name} reconstruction...")
    
    # MSE and MAE
    mse = mean_squared_error(orig.flatten(), recon.flatten())
    mae = mean_absolute_error(orig.flatten(), recon.flatten())
    
    # R2 score
    r2_mean = r2_score(orig, recon)
    r2_flat = r2_score(orig.flatten(), recon.flatten())
    
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
        'r2_mean': r2_mean,
        'r2_flat': r2_flat,
        'cosine_similarity': cos_sim,
        'pearson_mean': pearson_mean,
        'pearson_all': pearson_all
    }


def plot_reconstruction_examples(orig, recon, run_id, model_type, n_examples=3):
    """Plot example reconstructions"""
    print(f"\nGenerating reconstruction plots ({n_examples} examples)...")
    
    indices = np.random.choice(len(orig), n_examples, replace=False)
    
    run_suffix = f"_{run_id}" if run_id else ""
    modality = "RNA" if model_type == "rna" else "DNA"
    
    for idx_num, idx in enumerate(indices):
        plt.figure(figsize=(10, 5))
        
        plt.plot(orig[idx], label=f"Original {modality}", alpha=0.7, linewidth=2)
        plt.plot(recon[idx], label=f"Reconstructed {modality}", alpha=0.7, linewidth=2)
        plt.title(f"{modality} Reconstruction (CVAE) - Sample {idx_num+1}")
        plt.xlabel("Feature index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'plots/{model_type}_cvae_reconstruction_example_{idx_num+1}{run_suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_examples} reconstruction examples to plots/")


def plot_correlation_distribution(metrics, run_id, model_type):
    """Plot Pearson correlation distribution"""
    print("\nGenerating correlation distribution plot...")
    
    run_suffix = f"_{run_id}" if run_id else ""
    modality = "RNA" if model_type == "rna" else "DNA"
    
    plt.figure(figsize=(8, 5))
    plt.hist(
        metrics['pearson_all'], 
        bins=30, 
        color='skyblue', 
        edgecolor='black', 
        alpha=0.7
    )
    plt.axvline(
        metrics['pearson_mean'], 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {metrics["pearson_mean"]:.3f}'
    )
    plt.title(f"Pearson r distribution ({modality} CVAE Reconstruction)")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'plots/{model_type}_cvae_correlation_distribution{run_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distribution plot to {filename}")


def print_results(metrics, model_type):
    """Print evaluation results"""
    modality = "RNA" if model_type == "rna" else "DNA"
    
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {modality} CVAE")
    print("="*60)
    
    print(f"\n===== {modality} Reconstruction Metrics =====")
    print(f"  MSE:                {metrics['mse']:.4f}")
    print(f"  MAE:                {metrics['mae']:.4f}")
    print(f"  R2 (mean):          {metrics['r2_mean']:.4f}")
    print(f"  R2 (flat):          {metrics['r2_flat']:.4f}")
    print(f"  Cosine similarity:  {metrics['cosine_similarity']:.4f}")
    print(f"  Mean Pearson r:     {metrics['pearson_mean']:.4f}")
    
    print("\n" + "="*60)


def save_results(metrics, run_id, model_type):
    """Save evaluation results to file"""
    run_suffix = f"_{run_id}" if run_id else ""
    modality = "RNA" if model_type == "rna" else "DNA"
    
    results = {
        'run_id': run_id if run_id else 'default',
        'model_type': f'{modality}CVAE',
        'reconstruction': {
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'r2_mean': float(metrics['r2_mean']),
            'r2_flat': float(metrics['r2_flat']),
            'cosine_similarity': float(metrics['cosine_similarity']),
            'pearson_mean': float(metrics['pearson_mean']),
        }
    }
    
    import json
    filename = f'plots/{model_type}_cvae_evaluation_results{run_suffix}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def evaluate_model(model_type):
    """Evaluate a single CVAE model"""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load model and data
    model, val_dataloader, run_id = load_model_and_data(model_type)
    
    if model is None:
        print(f"\n⚠ Skipping {model_type.upper()} CVAE evaluation (model not found)")
        return
    
    # Reconstruct all samples
    orig, recon = reconstruct_all_samples(model, val_dataloader, model_type)
    
    print(f"Total validation samples: {len(orig)}")
    
    # Compute metrics
    modality = "RNA" if model_type == "rna" else "DNA"
    metrics = compute_metrics(orig, recon, modality)
    
    # Generate plots
    plot_reconstruction_examples(orig, recon, run_id, model_type, n_examples=3)
    plot_correlation_distribution(metrics, run_id, model_type)
    
    # Print and save results
    print_results(metrics, model_type)
    save_results(metrics, run_id, model_type)
    
    print(f"\n{modality} CVAE evaluation complete! All results saved to plots/")


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("CVAE EVALUATION PIPELINE")
    print("="*60)
    
    # Evaluate RNA CVAE
    print("\n" + "="*60)
    print("EVALUATING RNA CVAE")
    print("="*60)
    evaluate_model("rna")
    
    # Evaluate DNA CVAE
    print("\n" + "="*60)
    print("EVALUATING DNA CVAE")
    print("="*60)
    evaluate_model("dna")
    
    print("\n" + "="*60)
    print("All CVAE evaluations complete!")
    print("="*60)


if __name__ == "__main__":
    main()
