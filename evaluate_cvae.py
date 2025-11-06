"""
Evaluation script for Conditional Multi-Modal VAE (CVAE)
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from scipy.stats import pearsonr
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models.cvae import ConditionalMultiModalVAE
from src.data import MultiModalDataset


def load_model_and_data():
    """Load trained CVAE model and validation data"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    n_sites = len(label_encoder.classes_)
    
    # Split data (same split as training)
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
    
    # Get dimensions from first batch
    sample_batch = next(iter(val_dataloader))
    input_dim_rna = sample_batch[0].shape[1]
    input_dim_dna = sample_batch[1].shape[1]
    
    # Load model
    model_path = 'checkpoints/best_cvae.pt'
    print(f"Loading CVAE model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    
    model = ConditionalMultiModalVAE(
        input_dim_a=input_dim_rna,
        input_dim_b=input_dim_dna,
        n_sites=n_sites,
        latent_dim=checkpoint.get('latent_dim', Config.LATENT_DIM),
        embed_dim=32
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, val_dataloader, label_encoder


def reconstruct_all_samples(model, dataloader):
    """Reconstruct all samples in the validation set using CVAE"""
    print("\nReconstructing all validation samples with CVAE...")
    
    all_orig_rna, all_recon_rna = [], []
    all_orig_beta, all_recon_beta = [], []
    all_orig_site, all_pred_site = [], []
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm = tpm.to(Config.DEVICE)
            beta_data = beta_data.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            
            # Cross-modal reconstructions (CVAE requires site labels)
            # RNA from DNA (+ site)
            recon_a_from_b, _, recon_c_from_b, _, _ = model(a=None, b=beta_data, site=site)
            # DNA from RNA (+ site)
            _, recon_b_from_a, recon_c_from_a, _, _ = model(a=tpm, b=None, site=site)
            
            all_orig_rna.append(tpm.cpu().numpy())
            all_recon_rna.append(recon_a_from_b.cpu().numpy())
            all_orig_beta.append(beta_data.cpu().numpy())
            all_recon_beta.append(recon_b_from_a.cpu().numpy())
            
            # Site classification
            all_orig_site.append(site.cpu().numpy())
            # Average predictions from both paths
            pred_site = (recon_c_from_a + recon_c_from_b) / 2
            all_pred_site.append(torch.argmax(pred_site, dim=1).cpu().numpy())
    
    # Concatenate all batches
    orig_rna = np.concatenate(all_orig_rna, axis=0)
    recon_rna = np.concatenate(all_recon_rna, axis=0)
    orig_beta = np.concatenate(all_orig_beta, axis=0)
    recon_beta = np.concatenate(all_recon_beta, axis=0)
    orig_site = np.concatenate(all_orig_site, axis=0)
    pred_site = np.concatenate(all_pred_site, axis=0)
    
    return orig_rna, recon_rna, orig_beta, recon_beta, orig_site, pred_site


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


def compute_classification_metrics(orig_site, pred_site, label_encoder):
    """Compute site classification metrics"""
    print("\nComputing site classification metrics...")
    
    accuracy = accuracy_score(orig_site, pred_site)
    
    print(f"\nSite Classification Accuracy: {accuracy:.4f}")
    
    # Get unique classes that actually appear in the validation set
    unique_classes = np.unique(np.concatenate([orig_site, pred_site]))
    target_names = [label_encoder.classes_[i] for i in unique_classes]
    
    print("\nClassification Report:")
    print(classification_report(
        orig_site, 
        pred_site,
        labels=unique_classes,
        target_names=target_names,
        zero_division=0
    ))
    
    return accuracy


def plot_reconstruction_examples(orig_rna, recon_rna, orig_beta, recon_beta, n_examples=3):
    """Plot example reconstructions"""
    print(f"\nGenerating reconstruction plots ({n_examples} examples)...")
    
    indices = np.random.choice(len(orig_rna), n_examples, replace=False)
    
    for idx_num, idx in enumerate(indices):
        plt.figure(figsize=(12, 4))
        
        # RNA reconstruction
        plt.subplot(1, 2, 1)
        plt.plot(orig_rna[idx], label="Original RNA", alpha=0.7)
        plt.plot(recon_rna[idx], label="Reconstructed from DNA (CVAE)", alpha=0.7)
        plt.title("RNA reconstructed from DNA methylation (CVAE)")
        plt.xlabel("Gene index")
        plt.ylabel("Expression (normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # DNA methylation reconstruction
        plt.subplot(1, 2, 2)
        plt.plot(orig_beta[idx], label="Original DNA methylation", alpha=0.7)
        plt.plot(recon_beta[idx], label="Reconstructed from RNA (CVAE)", alpha=0.7)
        plt.title("DNA methylation reconstructed from RNA (CVAE)")
        plt.xlabel("Probe index")
        plt.ylabel("Beta value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/cvae_reconstruction_example_{idx_num+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_examples} reconstruction examples to plots/")


def plot_correlation_distributions(metrics_rna, metrics_beta):
    """Plot Pearson correlation distributions"""
    print("\nGenerating correlation distribution plots...")
    
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
    plt.title("Pearson r distribution (RNA ← DNA) - CVAE")
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
    plt.title("Pearson r distribution (DNA ← RNA) - CVAE")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'plots/cvae_correlation_distributions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distribution plots to {filename}")


def plot_confusion_matrix(orig_site, pred_site, label_encoder):
    """Plot confusion matrix for site classification"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    print("\nGenerating confusion matrix...")
    
    # Get unique classes that actually appear in the data
    unique_classes = np.unique(np.concatenate([orig_site, pred_site]))
    class_names = [label_encoder.classes_[i] for i in unique_classes]
    
    cm = confusion_matrix(orig_site, pred_site, labels=unique_classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Site Classification Confusion Matrix (CVAE)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/cvae_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved confusion matrix to plots/cvae_confusion_matrix.png")


def print_results(metrics_rna, metrics_beta, accuracy):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("CVAE EVALUATION RESULTS")
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
    
    print("\n===== Site Classification =====")
    print(f"  Accuracy:           {accuracy:.4f}")
    
    print("\n" + "="*60)


def save_results(metrics_rna, metrics_beta, accuracy):
    """Save evaluation results to file"""
    results = {
        'model': 'CVAE',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        },
        'site_classification': {
            'accuracy': float(accuracy)
        }
    }
    
    import json
    filename = 'plots/cvae_evaluation_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    """Main evaluation pipeline for CVAE"""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load model and data
    model, val_dataloader, label_encoder = load_model_and_data()
    
    # Reconstruct all samples
    orig_rna, recon_rna, orig_beta, recon_beta, orig_site, pred_site = reconstruct_all_samples(
        model, val_dataloader
    )
    
    print(f"\nTotal validation samples: {len(orig_rna)}")
    
    # Compute metrics
    metrics_rna = compute_metrics(orig_rna, recon_rna, "RNA from DNA")
    metrics_beta = compute_metrics(orig_beta, recon_beta, "DNA from RNA")
    accuracy = compute_classification_metrics(orig_site, pred_site, label_encoder)
    
    # Generate plots
    plot_reconstruction_examples(orig_rna, recon_rna, orig_beta, recon_beta, n_examples=3)
    plot_correlation_distributions(metrics_rna, metrics_beta)
    plot_confusion_matrix(orig_site, pred_site, label_encoder)
    
    # Print and save results
    print_results(metrics_rna, metrics_beta, accuracy)
    save_results(metrics_rna, metrics_beta, accuracy)
    
    print("\n✓ CVAE evaluation complete! All results saved to plots/")


if __name__ == "__main__":
    main()

