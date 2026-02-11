"""
Reconstruction of unmatched data using trained directional VAE models

This script:
1. Loads RNA-only samples (without matching DNA) and reconstructs DNA using RNA2DNAVAE
2. Loads DNA-only samples (without matching RNA) and reconstructs RNA using DNA2RNAVAE
3. Saves the reconstructed data for future use
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from src.config import Config
from src.models import RNA2DNAVAE, DNA2RNAVAE


def get_run_ids():
    """Get run IDs for both directional models"""
    rna2dna_run_id = None
    dna2rna_run_id = None
    
    if os.path.exists('latest_rna2dna_run_id.txt'):
        with open('latest_rna2dna_run_id.txt', 'r') as f:
            rna2dna_run_id = f.read().strip()
    
    if os.path.exists('latest_dna2rna_run_id.txt'):
        with open('latest_dna2rna_run_id.txt', 'r') as f:
            dna2rna_run_id = f.read().strip()
    
    return rna2dna_run_id, dna2rna_run_id


def load_models(label_encoder):
    """Load trained directional models"""
    rna2dna_run_id, dna2rna_run_id = get_run_ids()
    n_sites = len(label_encoder.classes_)

    # Runtime overrides (useful on Kaggle and for ad-hoc runs).
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None and "DEVICE" not in os.environ:
        Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif "DEVICE" in os.environ:
        Config.DEVICE = torch.device(os.environ["DEVICE"])

    Config.INPUT_DIM_A = int(os.getenv("INPUT_DIM_A", Config.INPUT_DIM_A))
    Config.INPUT_DIM_B = int(os.getenv("INPUT_DIM_B", Config.INPUT_DIM_B))
    Config.LATENT_DIM = int(os.getenv("LATENT_DIM", Config.LATENT_DIM))
    
    # Load RNA2DNAVAE model
    rna2dna_model = None
    if rna2dna_run_id:
        model_filename = f'best_rna2dna_{rna2dna_run_id}.pt'
        print(f"Loading RNA2DNAVAE from run: {rna2dna_run_id}")
        rna2dna_model = RNA2DNAVAE(
            Config.INPUT_DIM_A, 
            Config.INPUT_DIM_B, 
            n_sites, 
            Config.LATENT_DIM
        ).to(Config.DEVICE)
        
        model_path = os.path.join(Config.CHECKPOINT_DIR, model_filename)
        if os.path.exists(model_path):
            rna2dna_model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            rna2dna_model.eval()
            print(f"✓ RNA2DNAVAE model loaded successfully")
        else:
            print(f"Warning: Model file {model_path} not found!")
            rna2dna_model = None
    else:
        print("Warning: No RNA2DNAVAE run ID found.")
    
    # Load DNA2RNAVAE model
    dna2rna_model = None
    if dna2rna_run_id:
        model_filename = f'best_dna2rna_{dna2rna_run_id}.pt'
        print(f"Loading DNA2RNAVAE from run: {dna2rna_run_id}")
        dna2rna_model = DNA2RNAVAE(
            Config.INPUT_DIM_A, 
            Config.INPUT_DIM_B, 
            n_sites, 
            Config.LATENT_DIM
        ).to(Config.DEVICE)
        
        model_path = os.path.join(Config.CHECKPOINT_DIR, model_filename)
        if os.path.exists(model_path):
            dna2rna_model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            dna2rna_model.eval()
            print(f"✓ DNA2RNAVAE model loaded successfully")
        else:
            print(f"Warning: Model file {model_path} not found!")
            dna2rna_model = None
    else:
        print("Warning: No DNA2RNAVAE run ID found.")
    
    return rna2dna_model, dna2rna_model, rna2dna_run_id, dna2rna_run_id


def reconstruct_dna_from_rna(rna2dna_model, rna_df, label_encoder):
    """
    Reconstruct DNA methylation from RNA expression data
    
    Args:
        rna2dna_model: Trained RNA2DNAVAE model
        rna_df: DataFrame with columns ['case_barcode', 'tpm_unstranded', 'primary_site']
        label_encoder: LabelEncoder for primary_site
        
    Returns:
        DataFrame with reconstructed DNA data
    """
    print("\n" + "="*80)
    print("RECONSTRUCTING DNA FROM RNA-ONLY SAMPLES")
    print("="*80)
    print(f"Number of RNA-only samples: {len(rna_df)}")
    
    # Prepare data
    rna_data = np.array(rna_df['tpm_unstranded'].tolist()).astype(np.float32)
    
    # Normalize RNA data (log1p as in training)
    rna_data = np.log1p(rna_data)
    
    # Encode primary sites
    site_labels = label_encoder.transform(rna_df['primary_site'])
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        torch.tensor(rna_data).float(),
        torch.tensor(site_labels).long()
    )
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Reconstruct DNA
    reconstructed_dna_batches = []
    with torch.no_grad():
        for rna, site in dataloader:
            rna = rna.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            
            # Predict DNA from RNA + site
            recon_dna, _, _ = rna2dna_model(rna=rna, site=site)
            reconstructed_dna_batches.append(recon_dna.cpu().numpy())
    
    reconstructed_dna = np.concatenate(reconstructed_dna_batches, axis=0)
    print(f"✓ Reconstructed DNA shape: {reconstructed_dna.shape}")
    
    # Create output DataFrame
    result_df = rna_df.copy()
    result_df['reconstructed_beta_value'] = list(reconstructed_dna)
    result_df['primary_site_encoded'] = site_labels
    
    return result_df


def reconstruct_rna_from_dna(dna2rna_model, dna_df, label_encoder):
    """
    Reconstruct RNA expression from DNA methylation data
    
    Since we don't have primary_site for DNA-only samples, we'll try:
    1. Without site information (site=None)
    2. Or iterate through all possible sites and average/pick best
    
    Args:
        dna2rna_model: Trained DNA2RNAVAE model
        dna_df: DataFrame with columns ['case_barcode', 'beta_value']
        label_encoder: LabelEncoder for primary_site
        
    Returns:
        DataFrame with reconstructed RNA data
    """
    print("\n" + "="*80)
    print("RECONSTRUCTING RNA FROM DNA-ONLY SAMPLES")
    print("="*80)
    print(f"Number of DNA-only samples: {len(dna_df)}")
    print("\nNote: DNA-only samples don't have primary_site information.")
    print("Trying reconstruction without site information (site=None)...")
    
    # Prepare data
    dna_data = np.array(dna_df['beta_value'].tolist()).astype(np.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(torch.tensor(dna_data).float())
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Reconstruct RNA without site information
    reconstructed_rna_batches = []
    with torch.no_grad():
        for (dna,) in dataloader:
            dna = dna.to(Config.DEVICE)
            
            # Predict RNA from DNA only (no site)
            recon_rna, _, _ = dna2rna_model(dna=dna, site=None)
            reconstructed_rna_batches.append(recon_rna.cpu().numpy())
    
    reconstructed_rna = np.concatenate(reconstructed_rna_batches, axis=0)
    print(f"✓ Reconstructed RNA shape: {reconstructed_rna.shape}")
    
    # Create output DataFrame
    result_df = dna_df.copy()
    result_df['reconstructed_tpm_unstranded'] = list(reconstructed_rna)
    
    return result_df


def save_reconstruction_stats(rna_reconstructed_df, dna_reconstructed_df, run_timestamp):
    """Save statistics about the reconstruction"""
    stats = {
        'timestamp': run_timestamp,
        'rna_only_samples': len(rna_reconstructed_df) if rna_reconstructed_df is not None else 0,
        'dna_only_samples': len(dna_reconstructed_df) if dna_reconstructed_df is not None else 0,
    }
    
    if rna_reconstructed_df is not None:
        stats['rna_only_primary_sites'] = rna_reconstructed_df['primary_site'].value_counts().to_dict()
    
    with open(f'data/reconstruction_stats_{run_timestamp}.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\n✓ Reconstruction statistics saved to: data/reconstruction_stats_{run_timestamp}.pkl")


def main():
    """Main reconstruction pipeline"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("UNMATCHED DATA RECONSTRUCTION")
    print("="*80)
    print(f"Run timestamp: {run_timestamp}\n")
    
    # Load label encoder
    print("Loading label encoder...")
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded ({len(label_encoder.classes_)} classes)")
    
    # Load models
    print("\nLoading trained models...")
    rna2dna_model, dna2rna_model, rna2dna_run_id, dna2rna_run_id = load_models(label_encoder)
    
    # Check if unmatched data files exist
    rna_only_path = 'data/rna_only_unmatched.pkl'
    dna_only_path = 'data/dna_only_unmatched.pkl'
    
    rna_reconstructed_df = None
    dna_reconstructed_df = None
    
    # Reconstruct DNA from RNA-only samples
    if os.path.exists(rna_only_path) and rna2dna_model is not None:
        print(f"\nLoading RNA-only samples from: {rna_only_path}")
        rna_only_df = pd.read_pickle(rna_only_path)
        
        # Filter to keep only samples with primary_site in the label encoder
        original_len = len(rna_only_df)
        rna_only_df = rna_only_df[rna_only_df['primary_site'].isin(label_encoder.classes_)]
        if len(rna_only_df) < original_len:
            print(f"  Filtered out {original_len - len(rna_only_df)} samples with unknown primary_site")
        
        if len(rna_only_df) > 0:
            rna_reconstructed_df = reconstruct_dna_from_rna(
                rna2dna_model, rna_only_df, label_encoder
            )
            
            # Save reconstructed data
            output_path = f'data/rna_with_reconstructed_dna_{run_timestamp}.pkl'
            rna_reconstructed_df.to_pickle(output_path)
            print(f"✓ Saved reconstructed data to: {output_path}")
        else:
            print("  No samples to reconstruct (all filtered out)")
    elif not os.path.exists(rna_only_path):
        print(f"\n⚠ RNA-only data file not found: {rna_only_path}")
        print("  Run scripts/prepare_data.py first to generate unmatched data.")
    elif rna2dna_model is None:
        print(f"\n⚠ RNA2DNAVAE model not available")
        print("  Run train_rna2dna.py first to train the model.")
    
    # Reconstruct RNA from DNA-only samples
    if os.path.exists(dna_only_path) and dna2rna_model is not None:
        print(f"\nLoading DNA-only samples from: {dna_only_path}")
        dna_only_df = pd.read_pickle(dna_only_path)
        
        if len(dna_only_df) > 0:
            dna_reconstructed_df = reconstruct_rna_from_dna(
                dna2rna_model, dna_only_df, label_encoder
            )
            
            # Save reconstructed data
            output_path = f'data/dna_with_reconstructed_rna_{run_timestamp}.pkl'
            dna_reconstructed_df.to_pickle(output_path)
            print(f"✓ Saved reconstructed data to: {output_path}")
        else:
            print("  No samples to reconstruct")
    elif not os.path.exists(dna_only_path):
        print(f"\n⚠ DNA-only data file not found: {dna_only_path}")
        print("  Run scripts/prepare_data.py first to generate unmatched data.")
    elif dna2rna_model is None:
        print(f"\n⚠ DNA2RNAVAE model not available")
        print("  Run train_dna2rna.py first to train the model.")
    
    # Save reconstruction statistics
    if rna_reconstructed_df is not None or dna_reconstructed_df is not None:
        save_reconstruction_stats(rna_reconstructed_df, dna_reconstructed_df, run_timestamp)
    
    # Print summary
    print("\n" + "="*80)
    print("RECONSTRUCTION SUMMARY")
    print("="*80)
    if rna_reconstructed_df is not None:
        print(f"\n✓ RNA→DNA reconstruction completed")
        print(f"  Samples: {len(rna_reconstructed_df)}")
        print(f"  Output: data/rna_with_reconstructed_dna_{run_timestamp}.pkl")
        print(f"  Primary sites distribution:")
        for site, count in rna_reconstructed_df['primary_site'].value_counts().head(10).items():
            print(f"    - {site}: {count}")
    else:
        print(f"\n✗ RNA→DNA reconstruction not performed")
    
    if dna_reconstructed_df is not None:
        print(f"\n✓ DNA→RNA reconstruction completed")
        print(f"  Samples: {len(dna_reconstructed_df)}")
        print(f"  Output: data/dna_with_reconstructed_rna_{run_timestamp}.pkl")
    else:
        print(f"\n✗ DNA→RNA reconstruction not performed")
    
    print("\n" + "="*80)
    print("Reconstruction complete!")
    print("="*80)


if __name__ == "__main__":
    main()

