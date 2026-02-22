"""
t-SNE and PCA visualization for different imputation methods (KNN, Mean)

This script performs dimensionality reduction on:
1. RNA-only samples with imputed DNA (using KNN and Mean imputation)
2. DNA-only samples with imputed RNA (using KNN and Mean imputation)

Visualizes the results with primary site labels where available.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import sys
from pathlib import Path

# Add project root to python path to allow importing from src
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config

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


def load_data():
    """Load training data and unmatched samples"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load processed training data
    print("\nLoading processed training data...")
    train_df = pd.read_pickle('data/processed_data.pkl')
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Load unmatched samples
    print("\nLoading unmatched samples...")
    rna_only_df = None
    dna_only_df = None
    
    if os.path.exists('data/rna_only_unmatched.pkl'):
        rna_only_df = pd.read_pickle('data/rna_only_unmatched.pkl')
        print(f"✓ Loaded {len(rna_only_df)} RNA-only samples")
    
    if os.path.exists('data/dna_only_unmatched.pkl'):
        dna_only_df = pd.read_pickle('data/dna_only_unmatched.pkl')
        print(f"✓ Loaded {len(dna_only_df)} DNA-only samples")
    
    # Load label encoder
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder loaded ({len(label_encoder.classes_)} classes)")
    
    return train_df, rna_only_df, dna_only_df, label_encoder


def prepare_features(df, use_original_rna=True, use_imputed_dna=True,
                     use_original_dna=False, use_imputed_rna=False,
                     rna_col='tpm_unstranded', dna_col='beta_value',
                     imputed_rna_col='imputed_tpm_unstranded', imputed_dna_col='imputed_beta_value'):
    """
    Prepare feature matrix from dataframe
    
    Args:
        df: DataFrame with data
        use_original_rna: Include original RNA data
        use_imputed_dna: Include imputed DNA data
        use_original_dna: Include original DNA data
        use_imputed_rna: Include imputed RNA data
        rna_col: Column name for RNA data
        dna_col: Column name for DNA data
        imputed_rna_col: Column name for imputed RNA data
        imputed_dna_col: Column name for imputed DNA data
    
    Returns:
        Feature matrix (numpy array)
    """
    features = []
    
    if use_original_rna and rna_col in df.columns:
        rna_data = np.array(df[rna_col].tolist()).astype(np.float32)
        # Already log-normalized in the file
        features.append(rna_data)
    
    if use_imputed_dna and imputed_dna_col in df.columns:
        dna_data = np.array(df[imputed_dna_col].tolist()).astype(np.float32)
        features.append(dna_data)
    
    if use_original_dna and dna_col in df.columns:
        dna_data = np.array(df[dna_col].tolist()).astype(np.float32)
        features.append(dna_data)
    
    if use_imputed_rna and imputed_rna_col in df.columns:
        rna_data = np.array(df[imputed_rna_col].tolist()).astype(np.float32)
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
                     figsize=(12, 10), marker_size=50, alpha=0.7):
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
                # Use short label if available
                if label_name in class_short_labels:
                    label_name = class_short_labels[label_name]
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


def apply_mean_imputation(train_df, rna_only_df, dna_only_df):
    """Apply mean imputation to unmatched samples"""
    print("\n" + "="*80)
    print("APPLYING MEAN IMPUTATION")
    print("="*80)
    
    # Prepare training data
    train_rna = np.array(train_df['tpm_unstranded'].tolist()).astype(np.float32)
    train_dna = np.array(train_df['beta_value'].tolist()).astype(np.float32)
    
    # Fit imputers on training data
    rna_imputer = SimpleImputer(strategy="mean")
    dna_imputer = SimpleImputer(strategy="mean")
    
    rna_imputer.fit(train_rna)
    dna_imputer.fit(train_dna)
    
    # Get mean vectors
    rna_mean = rna_imputer.statistics_.astype(np.float32)
    dna_mean = dna_imputer.statistics_.astype(np.float32)
    
    # Apply to RNA-only samples (impute DNA)
    rna_only_imputed = rna_only_df.copy() if rna_only_df is not None else None
    if rna_only_imputed is not None:
        rna_only_imputed['imputed_beta_value'] = [dna_mean] * len(rna_only_imputed)
        print(f"✓ Applied mean imputation to {len(rna_only_imputed)} RNA-only samples")
    
    # Apply to DNA-only samples (impute RNA)
    dna_only_imputed = dna_only_df.copy() if dna_only_df is not None else None
    if dna_only_imputed is not None:
        # Need to log-normalize the mean RNA vector
        rna_mean_log = np.log1p(rna_mean)
        dna_only_imputed['imputed_tpm_unstranded'] = [rna_mean_log] * len(dna_only_imputed)
        print(f"✓ Applied mean imputation to {len(dna_only_imputed)} DNA-only samples")
    
    return rna_only_imputed, dna_only_imputed


def apply_knn_imputation(train_df, rna_only_df, dna_only_df, n_neighbors=5):
    """Apply KNN imputation to unmatched samples"""
    print("\n" + "="*80)
    print(f"APPLYING KNN IMPUTATION (k={n_neighbors})")
    print("="*80)
    
    # Prepare training data
    train_rna = np.array(train_df['tpm_unstranded'].tolist()).astype(np.float32)
    train_dna = np.array(train_df['beta_value'].tolist()).astype(np.float32)
    
    # Apply to RNA-only samples (impute DNA from RNA)
    rna_only_imputed = rna_only_df.copy() if rna_only_df is not None else None
    if rna_only_imputed is not None:
        print(f"  Imputing DNA for {len(rna_only_imputed)} RNA-only samples...")
        rna_val = np.array(rna_only_imputed['tpm_unstranded'].tolist()).astype(np.float32)
        
        # Fit KNN: DNA = f(RNA)
        knn_dna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        knn_dna.fit(train_rna, train_dna)
        dna_imputed = knn_dna.predict(rna_val)
        
        rna_only_imputed['imputed_beta_value'] = list(dna_imputed)
        print(f"✓ Applied KNN imputation to {len(rna_only_imputed)} RNA-only samples")
    
    # Apply to DNA-only samples (impute RNA from DNA)
    dna_only_imputed = dna_only_df.copy() if dna_only_df is not None else None
    if dna_only_imputed is not None:
        print(f"  Imputing RNA for {len(dna_only_imputed)} DNA-only samples...")
        dna_val = np.array(dna_only_imputed['beta_value'].tolist()).astype(np.float32)
        
        # Fit KNN: RNA = f(DNA)
        knn_rna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        knn_rna.fit(train_dna, train_rna)
        rna_imputed = knn_rna.predict(dna_val)
        
        # Log-normalize
        rna_imputed_log = np.log1p(rna_imputed)
        dna_only_imputed['imputed_tpm_unstranded'] = list(rna_imputed_log)
        print(f"✓ Applied KNN imputation to {len(dna_only_imputed)} DNA-only samples")
    
    return rna_only_imputed, dna_only_imputed


def analyze_samples(df, label_encoder, run_timestamp, method_name, sample_type):
    """Analyze samples with imputed data"""
    print("\n" + "="*80)
    print(f"ANALYZING {sample_type.upper()} SAMPLES ({method_name})")
    print("="*80)
    print(f"Number of samples: {len(df)}")
    
    # Filter to keep only samples with primary_site in the label encoder (if primary_site exists)
    if 'primary_site' in df.columns:
        original_len = len(df)
        df = df[df['primary_site'].isin(label_encoder.classes_)].copy()
        if len(df) < original_len:
            print(f"  Filtered out {original_len - len(df)} samples with primary_site not in label_encoder")
        if len(df) == 0:
            print("\n⚠ No samples with valid primary_site found")
            return
    
    # Prepare features
    if sample_type == 'RNA-only':
        features = prepare_features(
            df,
            use_original_rna=True,
            use_imputed_dna=True,
            imputed_dna_col='imputed_beta_value'
        )
    else:  # DNA-only
        features = prepare_features(
            df,
            use_original_dna=True,
            use_imputed_rna=True,
            imputed_rna_col='imputed_tpm_unstranded'
        )
    
    if features is None:
        print("⚠ Could not prepare features")
        return
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Get primary site labels
    site_labels = None
    if 'primary_site_encoded' in df.columns:
        site_labels = df['primary_site_encoded'].values
    elif 'primary_site' in df.columns:
        site_labels = label_encoder.transform(df['primary_site'])
        print(f"\nPrimary site distribution:")
        for site, count in df['primary_site'].value_counts().items():
            print(f"  {site}: {count}")
    else:
        print("\n⚠ No primary site labels found")
        return
    
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
        
        pca_title = f'PCA: {sample_type} samples ({method_name} imputation)\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nPCA Silh: {pca_score:.3f} | PCA NH: {pca_nh:.3f}'
        tsne_title = f't-SNE: {sample_type} samples ({method_name} imputation)\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nt-SNE Silh: {tsne_score:.3f} | t-SNE NH: {tsne_nh:.3f}'
        
        print(f"\n  Original features - Silhouette: {orig_score:.3f}, NH: {orig_nh:.3f}")
        print(f"  PCA features - Silhouette: {pca_score:.3f}, NH: {pca_nh:.3f}")
        print(f"  t-SNE features - Silhouette: {tsne_score:.3f}, NH: {tsne_nh:.3f}")
    else:
        pca_title = f'PCA: {sample_type} samples ({method_name} imputation)\n(colored by primary site)'
        tsne_title = f't-SNE: {sample_type} samples ({method_name} imputation)\n(colored by primary site)'
        
    # Plot PCA
    plot_clusters_2d(
        pca_features,
        site_labels,
        pca_title,
        f'plots/clustering/{sample_type.lower().replace("-", "_")}_pca_{method_name.lower().replace(" ", "_")}_{run_timestamp}.png',
        label_encoder=label_encoder
    )
    
    # Plot t-SNE
    plot_clusters_2d(
        tsne_features,
        site_labels,
        tsne_title,
        f'plots/clustering/{sample_type.lower().replace("-", "_")}_tsne_{method_name.lower().replace(" ", "_")}_{run_timestamp}.png',
        label_encoder=label_encoder
    )
    
    return features, pca_features, tsne_features


def main():
    """Main visualization pipeline"""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("DIMENSIONALITY REDUCTION VISUALIZATION FOR IMPUTATION METHODS")
    print("="*80)
    print(f"Run timestamp: {run_timestamp}\n")
    
    # Load data
    train_df, rna_only_df, dna_only_df, label_encoder = load_data()
    
    if rna_only_df is None and dna_only_df is None:
        print("\n⚠ No unmatched samples found!")
        print("  Run scripts/prepare_data.py first to generate unmatched data.")
        return
    
    # Create output directory
    os.makedirs('plots/clustering', exist_ok=True)
    
    # Apply Mean imputation
    rna_only_mean, dna_only_mean = apply_mean_imputation(train_df, rna_only_df, dna_only_df)
    
    # Apply KNN imputation
    rna_only_knn, dna_only_knn = apply_knn_imputation(train_df, rna_only_df, dna_only_df, n_neighbors=5)
    
    # Analyze with Mean imputation
    if rna_only_mean is not None and len(rna_only_mean) > 0:
        analyze_samples(rna_only_mean, label_encoder, run_timestamp, "Mean", "RNA-only")
    
    if dna_only_mean is not None and len(dna_only_mean) > 0:
        # For DNA-only, we need to check if primary_site is available
        if 'primary_site' not in dna_only_mean.columns:
            print("\n⚠ DNA-only samples don't have primary_site information")
            print("  Skipping visualization for DNA-only samples with Mean imputation")
        else:
            analyze_samples(dna_only_mean, label_encoder, run_timestamp, "Mean", "DNA-only")
    
    # Analyze with KNN imputation
    if rna_only_knn is not None and len(rna_only_knn) > 0:
        analyze_samples(rna_only_knn, label_encoder, run_timestamp, "KNN", "RNA-only")
    
    if dna_only_knn is not None and len(dna_only_knn) > 0:
        # For DNA-only, we need to check if primary_site is available
        if 'primary_site' not in dna_only_knn.columns:
            print("\n⚠ DNA-only samples don't have primary_site information")
            print("  Skipping visualization for DNA-only samples with KNN imputation")
        else:
            analyze_samples(dna_only_knn, label_encoder, run_timestamp, "KNN", "DNA-only")
    
    print("\n" + "="*80)
    print("Visualization analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
