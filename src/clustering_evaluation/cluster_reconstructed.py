"""
t-SNE and PCA visualization of reconstructed unmatched data

This script performs dimensionality reduction on:
1. RNA-only samples with reconstructed DNA
2. DNA-only samples with reconstructed RNA

Visualizes the results with primary site labels where available.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import glob


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


def find_latest_reconstruction_files():
    """Find the most recent reconstruction files"""
    rna_files = glob.glob('data/rna_with_reconstructed_dna_*.pkl')
    dna_files = glob.glob('data/dna_with_reconstructed_rna_*.pkl')
    
    rna_file = max(rna_files, key=os.path.getctime) if rna_files else None
    dna_file = max(dna_files, key=os.path.getctime) if dna_files else None
    
    return rna_file, dna_file


def load_reconstructed_data():
    """Load reconstructed data files"""
    print("="*80)
    print("LOADING RECONSTRUCTED DATA")
    print("="*80)
    
    rna_file, dna_file = find_latest_reconstruction_files()
    
    rna_reconstructed_df = None
    dna_reconstructed_df = None
    
    if rna_file:
        print(f"\nLoading RNA with reconstructed DNA from: {rna_file}")
        rna_reconstructed_df = pd.read_pickle(rna_file)
        print(f"✓ Loaded {len(rna_reconstructed_df)} RNA-only samples")
        print(f"  Columns: {list(rna_reconstructed_df.columns)}")
    else:
        print("\n⚠ No RNA reconstruction files found")
    
    if dna_file:
        print(f"\nLoading DNA with reconstructed RNA from: {dna_file}")
        dna_reconstructed_df = pd.read_pickle(dna_file)
        print(f"✓ Loaded {len(dna_reconstructed_df)} DNA-only samples")
        print(f"  Columns: {list(dna_reconstructed_df.columns)}")
    else:
        print("\n⚠ No DNA reconstruction files found")
    
    return rna_reconstructed_df, dna_reconstructed_df


def prepare_features(df, use_original_rna=True, use_reconstructed_dna=True, 
                     use_original_dna=False, use_reconstructed_rna=False):
    """
    Prepare feature matrix from dataframe
    
    Args:
        df: DataFrame with data
        use_original_rna: Include original RNA data
        use_reconstructed_dna: Include reconstructed DNA data
        use_original_dna: Include original DNA data
        use_reconstructed_rna: Include reconstructed RNA data
    
    Returns:
        Feature matrix (numpy array)
    """
    features = []
    
    if use_original_rna and 'tpm_unstranded' in df.columns:
        rna_data = np.array(df['tpm_unstranded'].tolist()).astype(np.float32)
        # Already log-normalized in the file
        features.append(rna_data)
    
    if use_reconstructed_dna and 'reconstructed_beta_value' in df.columns:
        dna_data = np.array(df['reconstructed_beta_value'].tolist()).astype(np.float32)
        features.append(dna_data)
    
    if use_original_dna and 'beta_value' in df.columns:
        dna_data = np.array(df['beta_value'].tolist()).astype(np.float32)
        features.append(dna_data)
    
    if use_reconstructed_rna and 'reconstructed_tpm_unstranded' in df.columns:
        rna_data = np.array(df['reconstructed_tpm_unstranded'].tolist()).astype(np.float32)
        features.append(rna_data)
    
    if len(features) == 0:
        return None
    
    return np.concatenate(features, axis=1)


def perform_dimensionality_reduction(features, method='tsne', n_components=2, random_state=42):
    """
    Perform dimensionality reduction
    
    Args:
        features: Feature matrix
        method: 'pca', 'tsne', or 'both'
        n_components: Number of components for output
        random_state: Random seed
    
    Returns:
        Reduced features (if method is 'pca' or 'tsne') or tuple (pca_features, tsne_features)
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    if method in ['pca', 'both']:
        print(f"\nPerforming PCA ({n_components} components)...")
        pca = PCA(n_components=n_components, random_state=random_state)
        pca_features = pca.fit_transform(features_scaled)
        explained_var = pca.explained_variance_ratio_
        print(f"  Explained variance: {explained_var}")
        print(f"  Total explained variance: {explained_var.sum():.4f}")
        results['pca'] = pca_features
    
    if method in ['tsne', 'both']:
        print(f"\nPerforming t-SNE ({n_components} components)...")
        # Use PCA for preprocessing if features are high-dimensional
        if features_scaled.shape[1] > 50:
            print(f"  Pre-processing with PCA (50 components)...")
            pca_pre = PCA(n_components=50, random_state=random_state)
            features_for_tsne = pca_pre.fit_transform(features_scaled)
        else:
            features_for_tsne = features_scaled
        
        tsne = TSNE(n_components=n_components, random_state=random_state, 
                   perplexity=min(30, len(features) - 1))
        tsne_features = tsne.fit_transform(features_for_tsne)
        print(f"  t-SNE completed")
        results['tsne'] = tsne_features
    
    if method == 'both':
        return results['pca'], results['tsne']
    else:
        return results[method]


def plot_clusters_2d(features_2d, labels, title, filename, label_encoder=None, 
                     figsize=(12, 10), marker_size=50, alpha=0.7, label_mapping=None):
    """
    Plot 2D clusters with labels
    
    Args:
        features_2d: 2D feature matrix
        labels: Labels for coloring (primary_site_encoded or cluster labels)
        title: Plot title
        filename: Output filename
        label_encoder: LabelEncoder to decode labels (optional)
        figsize: Figure size
        marker_size: Marker size
        alpha: Marker transparency
    """
    plt.figure(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    
    # Plot each cluster/site
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        # Get label name if encoder is provided
        if label_encoder is not None and label >= 0:
            try:
                label_name = label_encoder.inverse_transform([int(label)])[0]
            except:
                label_name = f"Site {int(label)}"
        else:
            label_name = f"Cluster {int(label)}" if label >= 0 else "Unknown"
        
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1],
            c=[colors[i]], 
            label=label_name,
            s=marker_size,
            alpha=alpha,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
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


def analyze_rna_only_samples(rna_df, label_encoder, run_timestamp):
    """Analyze RNA-only samples with reconstructed DNA"""
    print("\n" + "="*80)
    print("ANALYZING RNA-ONLY SAMPLES (with reconstructed DNA)")
    print("="*80)
    print(f"Number of samples: {len(rna_df)}")
    
    # Prepare features: Original RNA + Reconstructed DNA
    features = prepare_features(
        rna_df, 
        use_original_rna=True, 
        use_reconstructed_dna=True
    )
    
    if features is None:
        print("⚠ Could not prepare features")
        return
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Get primary site labels
    if 'primary_site_encoded' in rna_df.columns:
        site_labels = rna_df['primary_site_encoded'].values
    elif 'primary_site' in rna_df.columns:
        site_labels = label_encoder.transform(rna_df['primary_site'])
    else:
        print("⚠ No primary site labels found")
        return
    
    print(f"\nPrimary site distribution:")
    for site, count in pd.Series(site_labels).value_counts().items():
        site_name = label_encoder.inverse_transform([int(site)])[0]
        print(f"  {site_name}: {count}")
    
    # Dimensionality reduction
    pca_features, tsne_features = perform_dimensionality_reduction(
        features, method='both', n_components=2
    )
    
    # Calculate Silhouette scores
    pca_score = None
    tsne_score = None
    if len(np.unique(site_labels)) > 1:
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        
        def calculate_nh(feat, lab, k=5):
            if len(feat) < k + 1: return 0.0
            nn = NearestNeighbors(n_neighbors=k+1).fit(feat)
            ind = nn.kneighbors(feat, return_distance=False)[:, 1:]
            lab = np.array(lab)
            return np.mean([np.mean(lab[i] == lab[idx]) for i, idx in enumerate(ind)])

        orig_score = silhouette_score(features, site_labels)
        pca_score = silhouette_score(pca_features, site_labels)
        tsne_score = silhouette_score(tsne_features, site_labels)
        orig_nh = calculate_nh(features, site_labels)
        pca_nh = calculate_nh(pca_features, site_labels)
        tsne_nh = calculate_nh(tsne_features, site_labels)
        
        pca_title = f'PCA: RNA-only samples with reconstructed DNA\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nPCA Silh: {pca_score:.3f} | PCA NH: {pca_nh:.3f}'
        tsne_title = f't-SNE: RNA-only samples with reconstructed DNA\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nt-SNE Silh: {tsne_score:.3f} | t-SNE NH: {tsne_nh:.3f}'
        
        print(f"\n  Original features - Silhouette: {orig_score:.3f}, NH: {orig_nh:.3f}")
        print(f"  PCA features - Silhouette: {pca_score:.3f}, NH: {pca_nh:.3f}")
        print(f"  t-SNE features - Silhouette: {tsne_score:.3f}, NH: {tsne_nh:.3f}")
    else:
        pca_title = f'PCA: RNA-only samples with reconstructed DNA\n(colored by primary site)'
        tsne_title = f't-SNE: RNA-only samples with reconstructed DNA\n(colored by primary site)'
        
    # Plot PCA
    plot_clusters_2d(
        pca_features,
        site_labels,
        pca_title,
        f'plots/clustering/rna_only_pca_by_site_{run_timestamp}.png',
        label_encoder=label_encoder
    )
    
    # Plot t-SNE
    plot_clusters_2d(
        tsne_features,
        site_labels,
        tsne_title,
        f'plots/clustering/rna_only_tsne_by_site_{run_timestamp}.png',
        label_encoder=label_encoder,
        label_mapping=class_short_labels
    )
    
    return features, pca_features, tsne_features


def analyze_dna_only_samples(dna_df, run_timestamp):
    """Analyze DNA-only samples with reconstructed RNA"""
    print("\n" + "="*80)
    print("ANALYZING DNA-ONLY SAMPLES (with reconstructed RNA)")
    print("="*80)
    print(f"Number of samples: {len(dna_df)}")
    
    # Prepare features: Original DNA + Reconstructed RNA
    features = prepare_features(
        dna_df,
        use_original_dna=True,
        use_reconstructed_rna=True
    )
    
    if features is None:
        print("⚠ Could not prepare features")
        return
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Dimensionality reduction
    pca_features, tsne_features = perform_dimensionality_reduction(
        features, method='both', n_components=2
    )
    
    # Since we don't have primary site for DNA-only, plot without color coding
    print("\nNote: Primary site information not available for DNA-only samples")
    
    # Create simple labels (all same) for plotting
    simple_labels = np.zeros(len(dna_df), dtype=int)
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                c='steelblue', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('PCA: DNA-only samples with reconstructed RNA', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'plots/clustering/dna_only_pca_{run_timestamp}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ PCA plot saved to: {filename}")
    plt.close()
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                c='steelblue', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title('t-SNE: DNA-only samples with reconstructed RNA', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'plots/clustering/dna_only_tsne_{run_timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE plot saved to: {filename}")
    plt.close()
    
    return features, pca_features, tsne_features


def create_summary_report(rna_df, dna_df, run_timestamp):
    """Create a summary report of the visualization analysis"""
    report_lines = [
        "="*80,
        "DIMENSIONALITY REDUCTION VISUALIZATION SUMMARY",
        "="*80,
        f"Timestamp: {run_timestamp}",
        ""
    ]
    
    if rna_df is not None:
        report_lines.extend([
            "RNA-ONLY SAMPLES (with reconstructed DNA):",
            f"  Total samples: {len(rna_df)}",
            f"  Primary sites: {rna_df['primary_site'].nunique() if 'primary_site' in rna_df.columns else 'N/A'}",
            ""
        ])
    
    if dna_df is not None:
        report_lines.extend([
            "DNA-ONLY SAMPLES (with reconstructed RNA):",
            f"  Total samples: {len(dna_df)}",
            f"  Note: Primary site information not available",
            ""
        ])
    
    report_lines.extend([
        "Generated plots:",
        "  - PCA visualizations",
        "  - t-SNE visualizations",
        "  - Colored by primary site (where available)",
        "",
        "All plots saved to: plots/clustering/",
        "="*80
    ])
    
    report_text = "\n".join(report_lines)
    
    # Print report
    print("\n" + report_text)
    
    # Save report
    os.makedirs('plots/clustering', exist_ok=True)
    with open(f'plots/clustering/summary_report_{run_timestamp}.txt', 'w') as f:
        f.write(report_text)
    print(f"\n✓ Summary report saved to: plots/clustering/summary_report_{run_timestamp}.txt")


def main():
    """Main visualization pipeline"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("DIMENSIONALITY REDUCTION VISUALIZATION OF RECONSTRUCTED DATA")
    print("="*80)
    print(f"Run timestamp: {run_timestamp}\n")
    
    # Load label encoder
    print("Loading label encoder...")
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded ({len(label_encoder.classes_)} classes)")
    
    # Load reconstructed data
    rna_df, dna_df = load_reconstructed_data()
    
    if rna_df is None and dna_df is None:
        print("\n⚠ No reconstructed data found!")
        print("  Run reconstruct_unmatched.py first to generate reconstructed data.")
        return
    
    # Create output directory
    os.makedirs('plots/clustering', exist_ok=True)
    
    # Analyze RNA-only samples
    if rna_df is not None and len(rna_df) > 0:
        analyze_rna_only_samples(rna_df, label_encoder, run_timestamp)
    
    # Analyze DNA-only samples
    if dna_df is not None and len(dna_df) > 0:
        analyze_dna_only_samples(dna_df, run_timestamp)
    
    # Create summary report
    create_summary_report(rna_df, dna_df, run_timestamp)
    
    print("\n" + "="*80)
    print("Visualization analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
