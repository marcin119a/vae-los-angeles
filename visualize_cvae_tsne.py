"""
Visualize latent representations from CVAE models using t-SNE
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config
from src.models import RNACVAE, DNACVAE
from src.data import MultiModalDataset

# Mapping of full class names to short labels for legend
class_short_labels = {
    "Hematopoietic and reticuloendothelial systems": "Hemato",
    "Bronchus and lung": "Lung",
    "Breast": "Breast",
    "Kidney": "Kidney",
    "Brain": "Brain",
    "Colon": "Colon",
    "Corpus uteri": "Corpus",
    "Skin": "Skin",
    "Prostate gland": "Prostate",
    "Stomach": "Stomach",
    "Bladder": "Bladder",
    "Liver and intrahepatic bile ducts": "Liver",
    "Pancreas": "Pancreas",
    "Ovary": "Ovary",
    "Uterus, NOS": "Uterus",
    "Cervix uteri": "Cervix",
    "Esophagus": "Esophagus",
    "Adrenal gland": "Adrenal",
    "Other and ill-defined sites": "Other",
    "Other and unspecified parts of tongue": "Tongue",
    "Connective, subcutaneous and other soft tissues": "Connective",
    "Larynx": "Larynx",
    "Rectum": "Rectum",
    "Other and ill-defined sites in lip, oral cavity and pharynx": "Oral/Pharynx"
}


def get_run_id(model_type):
    """Get the run ID from latest training"""
    filename = f'latest_{model_type}_cvae_run_id.txt'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read().strip()
    return None


def load_model(model_type, n_sites):
    """Load trained CVAE model"""
    run_id = get_run_id(model_type)
    
    if not run_id:
        print(f"Warning: No {model_type.upper()} CVAE run ID found. Please train the model first.")
        return None, None
    
    if model_type == "rna":
        model_filename = f'best_rna_cvae_{run_id}.pt'
        model = RNACVAE(
            Config.INPUT_DIM_A,  # RNA dimension
            n_sites, 
            Config.LATENT_DIM
        ).to(Config.DEVICE)
    elif model_type == "dna":
        model_filename = f'best_dna_cvae_{run_id}.pt'
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
        print(f"✓ {model_type.upper()} CVAE model loaded from {model_path}")
        return model, run_id
    else:
        print(f"Error: Model file not found at {model_path}")
        return None, None


def extract_latent_representations(model, dataloader, model_type):
    """
    Extract latent representations (mu) from CVAE encoder
    
    Args:
        model: Trained CVAE model
        dataloader: DataLoader with data
        model_type: "rna" or "dna"
    
    Returns:
        latent_reps: numpy array of latent representations (n_samples, latent_dim)
        site_labels: numpy array of primary site labels
    """
    print(f"\nExtracting latent representations from {model_type.upper()} CVAE...")
    
    latent_reps = []
    site_labels = []
    
    with torch.no_grad():
        for tpm, beta_data, site in dataloader:
            tpm = tpm.to(Config.DEVICE)
            beta_data = beta_data.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            
            if model_type == "rna":
                # Get site embedding
                site_emb = model.site_embedding(site)
                # Create condition
                condition = torch.cat([site_emb, tpm], dim=1)
                # Encoder forward pass
                encoder_input = torch.cat([tpm, condition], dim=1)
                h = model.encoder(encoder_input)
                mu = model.fc_mu(h)
            else:  # dna
                # Flatten DNA
                dna_flat = beta_data.view(beta_data.size(0), -1)
                # Get site embedding
                site_emb = model.site_embedding(site)
                # Create condition
                condition = torch.cat([site_emb, dna_flat], dim=1)
                # Encoder forward pass
                encoder_input = torch.cat([dna_flat, condition], dim=1)
                h = model.encoder(encoder_input)
                mu = model.fc_mu(h)
            
            latent_reps.append(mu.cpu().numpy())
            site_labels.append(site.cpu().numpy())
    
    latent_reps = np.concatenate(latent_reps, axis=0)
    site_labels = np.concatenate(site_labels, axis=0)
    
    print(f"  Extracted {len(latent_reps)} latent representations")
    print(f"  Latent dimension: {latent_reps.shape[1]}")
    
    return latent_reps, site_labels


def perform_tsne(latent_reps, n_components=2, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction
    
    Args:
        latent_reps: Latent representations (n_samples, latent_dim)
        n_components: Number of t-SNE components (default: 2)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
    
    Returns:
        tsne_embedding: t-SNE embedding (n_samples, n_components)
    """
    print(f"\nPerforming t-SNE (perplexity={perplexity})...")
    
    # Standardize features
    scaler = StandardScaler()
    latent_reps_scaled = scaler.fit_transform(latent_reps)
    
    # Use PCA for preprocessing if features are high-dimensional
    if latent_reps_scaled.shape[1] > 50:
        print(f"  Pre-processing with PCA (50 components)...")
        pca_pre = PCA(n_components=50, random_state=random_state)
        latent_reps_for_tsne = pca_pre.fit_transform(latent_reps_scaled)
    else:
        latent_reps_for_tsne = latent_reps_scaled
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(latent_reps) - 1),
        random_state=random_state,
        verbose=1
    )
    tsne_embedding = tsne.fit_transform(latent_reps_for_tsne)
    
    print(f"  t-SNE completed")
    
    return tsne_embedding


def plot_tsne(tsne_embedding, site_labels, title, filename, label_encoder=None,
              figsize=(14, 10), marker_size=50, alpha=0.7):
    """
    Plot t-SNE visualization colored by primary site
    
    Args:
        tsne_embedding: t-SNE embedding (n_samples, 2)
        site_labels: Primary site labels
        title: Plot title
        filename: Output filename
        label_encoder: LabelEncoder to decode labels
        figsize: Figure size
        marker_size: Marker size
        alpha: Marker transparency
    """
    plt.figure(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(site_labels)
    n_labels = len(unique_labels)
    
    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    
    # Plot each site
    for i, label in enumerate(unique_labels):
        mask = site_labels == label
        
        # Get label name if encoder is provided
        if label_encoder is not None and label >= 0:
            try:
                label_name = label_encoder.inverse_transform([int(label)])[0]
                # Use short label if available
                if label_name in class_short_labels:
                    label_name = class_short_labels[label_name]
            except:
                label_name = f"Site {int(label)}"
        else:
            label_name = f"Site {int(label)}" if label >= 0 else "Unknown"
        
        plt.scatter(
            tsne_embedding[mask, 0],
            tsne_embedding[mask, 1],
            c=[colors[i]],
            label=label_name,
            s=marker_size,
            alpha=alpha,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Place legend outside plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, fontsize=9, ncol=1)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {filename}")
    plt.close()


def visualize_model(model_type, dataloader, label_encoder, run_id, output_dir):
    """Visualize latent space for a single CVAE model"""
    n_sites = len(label_encoder.classes_)
    
    # Load model
    model, model_run_id = load_model(model_type, n_sites)
    if model is None:
        return None
    
    # Use model_run_id if available, otherwise use provided run_id
    effective_run_id = model_run_id if model_run_id else run_id
    
    # Extract latent representations
    latent_reps, site_labels = extract_latent_representations(model, dataloader, model_type)
    
    # Perform t-SNE
    tsne_embedding = perform_tsne(latent_reps, perplexity=30)
    
    # Create visualization
    modality = "RNA" if model_type == "rna" else "DNA"
    title = f't-SNE: {modality} CVAE Latent Space\n(colored by primary site)'
    filename = os.path.join(
        output_dir,
        f'{model_type}_cvae_tsne_{effective_run_id if effective_run_id else "default"}.png'
    )
    
    plot_tsne(tsne_embedding, site_labels, title, filename, label_encoder)
    
    return tsne_embedding, latent_reps


def main():
    """Main visualization pipeline"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'plots/cvae_tsne'
    
    print("="*80)
    print("CVAE LATENT SPACE VISUALIZATION (t-SNE)")
    print("="*80)
    print(f"Run timestamp: {run_timestamp}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Data shape: {merged_df.shape}")
    print(f"Number of primary sites: {len(label_encoder.classes_)}")
    
    # Create dataset (use all data or validation split)
    use_all_data = input("\nUse all data for visualization? (y/n, default=n): ").strip().lower()
    
    if use_all_data == 'y':
        dataset = MultiModalDataset(merged_df)
        print("Using all data for visualization")
    else:
        _, val_df = train_test_split(
            merged_df, 
            test_size=Config.TRAIN_TEST_SPLIT, 
            random_state=Config.RANDOM_SEED
        )
        dataset = MultiModalDataset(val_df)
        print(f"Using validation split: {len(val_df)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    # Visualize RNA CVAE
    print("\n" + "="*80)
    print("VISUALIZING RNA CVAE")
    print("="*80)
    rna_tsne, rna_latent = visualize_model("rna", dataloader, label_encoder, run_timestamp, output_dir)
    
    # Visualize DNA CVAE
    print("\n" + "="*80)
    print("VISUALIZING DNA CVAE")
    print("="*80)
    dna_tsne, dna_latent = visualize_model("dna", dataloader, label_encoder, run_timestamp, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    if rna_tsne is not None:
        print(f"✓ RNA CVAE t-SNE visualization saved")
    if dna_tsne is not None:
        print(f"✓ DNA CVAE t-SNE visualization saved")
    print(f"\nAll plots saved to: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
