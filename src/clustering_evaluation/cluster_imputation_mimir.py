"""
t-SNE and PCA visualization for different imputation methods (KNN, Mean)

This script performs dimensionality reduction on:
1. RNA-only samples with imputed DNA (using KNN and Mean imputation)
2. DNA-only samples with imputed RNA (using KNN and Mean imputation)

Visualizes the results with primary site labels where available.
"""
import os
import sys
from pathlib import Path

# Add project root to python path to allow importing from src
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-ready seaborn theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Global unified color palette mapping for imputation methods
METHOD_COLORS = {
    'Mean': '#8da0cb',            # Muted blue
    'KNN': '#fc8d62',             # Muted orange
    'Conditioned KNN': '#e78ac3', # Muted pink
    'VAE': '#a6d854',             # Muted green
    'MIMIR': '#ffd92f'            # Muted yellow
}

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from src.models.conditioned_knn import ConditionedKNeighborsRegressor
from datetime import datetime

import sys
mimir_path = os.path.join(project_root, 'MIMIR')
if mimir_path not in sys.path:
    sys.path.append(mimir_path)

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


def apply_conditioned_knn_imputation(train_df, rna_only_df, dna_only_df, label_encoder, n_neighbors=5):
    """Apply Conditioned KNN imputation to unmatched samples"""
    print("\n" + "="*80)
    print(f"APPLYING CONDITIONED KNN IMPUTATION (k={n_neighbors})")
    print("="*80)
    
    # Prepare training data. Conditioned KNN expects the last column to be the primary site index.
    train_rna = np.array(train_df['tpm_unstranded'].tolist()).astype(np.float32)
    train_dna = np.array(train_df['beta_value'].tolist()).astype(np.float32)
    train_sites = train_df['primary_site_encoded'].values[:, np.newaxis]
    
    train_rna_cond = np.hstack((train_rna, train_sites))
    train_dna_cond = np.hstack((train_dna, train_sites))
    
    # Apply to RNA-only samples (impute DNA from RNA)
    rna_only_imputed = None
    if rna_only_df is not None:
        if 'primary_site' in rna_only_df.columns:
            # Filter samples with known primary sites
            valid_rna = rna_only_df[rna_only_df['primary_site'].isin(label_encoder.classes_)].copy()
            if len(valid_rna) > 0:
                print(f"  Imputing DNA for {len(valid_rna)} RNA-only samples with known primary site...")
                
                # Get site encoded
                if 'primary_site_encoded' in valid_rna.columns:
                    valid_sites = valid_rna['primary_site_encoded'].values
                else:
                    valid_sites = label_encoder.transform(valid_rna['primary_site'])
                
                rna_val = np.array(valid_rna['tpm_unstranded'].tolist()).astype(np.float32)
                rna_val_cond = np.hstack((rna_val, valid_sites[:, np.newaxis]))
                
                # Fit KNN: DNA = f(RNA, site)
                knn_dna = ConditionedKNeighborsRegressor(n_neighbors=n_neighbors)
                knn_dna.fit(train_rna_cond, train_dna)
                dna_imputed = knn_dna.predict(rna_val_cond)
                
                valid_rna['imputed_beta_value'] = list(dna_imputed)
                rna_only_imputed = valid_rna
                print(f"✓ Applied Conditioned KNN imputation to {len(valid_rna)} RNA-only samples")
            else:
                print("⚠ No RNA-only samples with valid primary_site found. Skipping Conditioned KNN.")
        else:
             print("⚠ RNA-only samples don't have primary_site information. Skipping Conditioned KNN.")

    # Apply to DNA-only samples (impute RNA from DNA)
    dna_only_imputed = None
    if dna_only_df is not None:
        if 'primary_site' in dna_only_df.columns:
            valid_dna = dna_only_df[dna_only_df['primary_site'].isin(label_encoder.classes_)].copy()
            if len(valid_dna) > 0:
                print(f"  Imputing RNA for {len(valid_dna)} DNA-only samples with known primary site...")
                
                # Get site encoded
                if 'primary_site_encoded' in valid_dna.columns:
                    valid_sites = valid_dna['primary_site_encoded'].values
                else:
                    valid_sites = label_encoder.transform(valid_dna['primary_site'])
                
                dna_val = np.array(valid_dna['beta_value'].tolist()).astype(np.float32)
                dna_val_cond = np.hstack((dna_val, valid_sites[:, np.newaxis]))
                
                # Fit KNN: RNA = f(DNA, site)
                knn_rna = ConditionedKNeighborsRegressor(n_neighbors=n_neighbors)
                knn_rna.fit(train_dna_cond, train_rna)
                rna_imputed = knn_rna.predict(dna_val_cond)
                
                # Log-normalize
                rna_imputed_log = np.log1p(rna_imputed)
                valid_dna['imputed_tpm_unstranded'] = list(rna_imputed_log)
                dna_only_imputed = valid_dna
                print(f"✓ Applied Conditioned KNN imputation to {len(valid_dna)} DNA-only samples")
            else:
                 print("⚠ No DNA-only samples with valid primary_site found. Skipping Conditioned KNN.")
        else:
             print("⚠ DNA-only samples don't have primary_site information. Skipping Conditioned KNN.")
    
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
            return None, None, None, None
    
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
        return None, None, None, None
    
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
        return None, None, None, None
    
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
        from sklearn.preprocessing import StandardScaler
        
        from src.clustering_evaluation.metrics_utils import calculate_neighborhood_hit
        
        features_scaled = StandardScaler().fit_transform(features)
        
        orig_score = silhouette_score(features_scaled, site_labels)
        orig_nh = calculate_neighborhood_hit(features_scaled, site_labels)
        
        pca_score = silhouette_score(pca_features, site_labels)
        tsne_score = silhouette_score(tsne_features, site_labels)
        
        pca_nh = calculate_neighborhood_hit(pca_features, site_labels)
        tsne_nh = calculate_neighborhood_hit(tsne_features, site_labels)
        
        pca_title = f'PCA: {sample_type} samples ({method_name} imputation)\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nPCA Silh: {pca_score:.3f} | NH: {pca_nh:.3f}'
        tsne_title = f't-SNE: {sample_type} samples ({method_name} imputation)\nOrig Silh: {orig_score:.3f} | Orig NH: {orig_nh:.3f}\nt-SNE Silh: {tsne_score:.3f} | NH: {tsne_nh:.3f}'
        
        print(f"\n  Original features - Silhouette: {orig_score:.3f}, NH: {orig_nh:.3f}")
        print(f"  PCA features - Silhouette: {pca_score:.3f}, NH: {pca_nh:.3f}")
        print(f"  t-SNE features - Silhouette: {tsne_score:.3f}, NH: {tsne_nh:.3f}")
        
        # Plot PCA
        plot_clusters_2d(
            pca_features,
            site_labels,
            pca_title,
            f'plots/clustering_mimir/{sample_type.lower().replace("-", "_")}_pca_{method_name.lower().replace(" ", "_")}_{run_timestamp}.png',
            label_encoder=label_encoder
        )
        
        # Plot t-SNE
        plot_clusters_2d(
            tsne_features,
            site_labels,
            tsne_title,
            f'plots/clustering_mimir/{sample_type.lower().replace("-", "_")}_tsne_{method_name.lower().replace(" ", "_")}_{run_timestamp}.png',
            label_encoder=label_encoder
        )
        
        metrics = {
            'orig_silh': orig_score,
            'orig_nh': orig_nh,
            'pca_silh': pca_score,
            'pca_nh': pca_nh,
            'tsne_silh': tsne_score,
            'tsne_nh': tsne_nh
        }
        return features, pca_features, tsne_features, metrics
    else:
        print(f"\n⚠ Not enough distinct primary site labels found ({len(np.unique(site_labels))} label(s)). Skipping plots.")
        return features, pca_features, tsne_features, None


def plot_metrics_summary(metrics_results, run_timestamp):
    df = pd.DataFrame(metrics_results)
    
    for sample_type in df['Sample Type'].unique():
        sub_df = df[df['Sample Type'] == sample_type]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette
        x = np.arange(len(sub_df))
        width = 0.25
        
        dim_colors = sns.color_palette("Set2", 3)
        
        axes[0].bar(x - width, sub_df['orig_silh'], width, label='Original', color=dim_colors[0], edgecolor='none')
        axes[0].bar(x, sub_df['pca_silh'], width, label='PCA', color=dim_colors[1], edgecolor='none')
        axes[0].bar(x + width, sub_df['tsne_silh'], width, label='t-SNE', color=dim_colors[2], edgecolor='none')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title(f'Silhouette Scores ({sample_type})')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(sub_df['Method'], rotation=45, ha='right')
        axes[0].legend()
        
        # Neighborhood Hits
        axes[1].bar(x - width, sub_df['orig_nh'], width, label='Original', color=dim_colors[0], edgecolor='none')
        axes[1].bar(x, sub_df['pca_nh'], width, label='PCA', color=dim_colors[1], edgecolor='none')
        axes[1].bar(x + width, sub_df['tsne_nh'], width, label='t-SNE', color=dim_colors[2], edgecolor='none')
        axes[1].set_ylabel('Neighborhood Hit')
        axes[1].set_title(f'Neighborhood Hits ({sample_type})')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sub_df['Method'], rotation=45, ha='right')
        axes[1].legend()
        
        plt.tight_layout()
        filename = f'plots/clustering_mimir/{sample_type.lower().replace("-", "_")}_metrics_summary_{run_timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Summary metrics plot saved to: {filename}")
        plt.close()

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
    os.makedirs('plots/clustering_mimir', exist_ok=True)
    
    # Load MIMIR imputed data
    rna_only_mimir = None
    dna_only_mimir = None
    if os.path.exists('data/rna_only_unmatched_mimir.pkl'):
        rna_only_mimir = pd.read_pickle('data/rna_only_unmatched_mimir.pkl')
        print(f"✓ Loaded {len(rna_only_mimir)} RNA-only samples with MIMIR imputation")
    if os.path.exists('data/dna_only_unmatched_mimir.pkl'):
        dna_only_mimir = pd.read_pickle('data/dna_only_unmatched_mimir.pkl')
        print(f"✓ Loaded {len(dna_only_mimir)} DNA-only samples with MIMIR imputation")
    
    import glob
    def find_latest_reconstruction_files():
        rna_files = glob.glob('data/rna_with_reconstructed_dna_*.pkl')
        dna_files = glob.glob('data/dna_with_reconstructed_rna_*.pkl')
        rna_file = max(rna_files, key=os.path.getctime) if rna_files else None
        dna_file = max(dna_files, key=os.path.getctime) if dna_files else None
        return rna_file, dna_file
        
    rna_vae_file, dna_vae_file = find_latest_reconstruction_files()
    rna_only_vae = None
    dna_only_vae = None
    if rna_vae_file:
        rna_only_vae = pd.read_pickle(rna_vae_file)
        # Rename original 'reconstructed_beta_value' to standardized 'imputed_beta_value' to track
        rna_only_vae = rna_only_vae.rename(columns={'reconstructed_beta_value': 'imputed_beta_value'})
        print(f"✓ Loaded {len(rna_only_vae)} RNA-only samples with VAE imputation")
    if dna_vae_file:
        dna_only_vae = pd.read_pickle(dna_vae_file)
        dna_only_vae = dna_only_vae.rename(columns={'reconstructed_tpm_unstranded': 'imputed_tpm_unstranded'})
        print(f"✓ Loaded {len(dna_only_vae)} DNA-only samples with VAE imputation")
        
    # Apply Mean imputation
    rna_only_mean, dna_only_mean = apply_mean_imputation(train_df, rna_only_df, dna_only_df)
    
    # -------------------------------------------------------------
    # NEW: Ground truth imputation accuracy evaluation
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATING IMPUTATION ACCURACY VS GROUND TRUTH")
    print("="*80)
    
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr
    
    # We use train_df which has both RNA and DNA for all samples to test accuracy.
    # Split into a new train and test set to evaluate KNN methods
    train_subset, test_subset = train_test_split(train_df, test_size=0.1, random_state=42)
    
    # Prepare true values
    true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
    true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)
    
    # We simulate "RNA-only" samples by extracting only RNA from the test subset
    rna_test_df = test_subset.drop(columns=['beta_value']).copy()

    acc_results = []
    
    def evaluate_reconstruction(true_vals, pred_vals, method, target_modality):
        # Flatten for global correlation/MSE
        t_flat = true_vals.flatten()
        p_flat = pred_vals.flatten()
        
        mse = np.mean((t_flat - p_flat)**2)
        pearson, _ = pearsonr(t_flat, p_flat)
        
        print(f"  {method} ({target_modality}): MSE = {mse:.4f}, Pearson r = {pearson:.4f}")
        acc_results.append({
            'Method': method,
            'Target': target_modality,
            'MSE': mse,
            'Pearson r': pearson
        })

    # 1. Evaluate Mean
    trn_dna_mean = np.mean(np.array(train_subset['beta_value'].tolist()).astype(np.float32), axis=0)
    mean_preds = np.tile(trn_dna_mean, (len(test_subset), 1))
    evaluate_reconstruction(true_dna_te, mean_preds, 'Mean', 'DNA')
    
    # 2. Evaluate KNN
    trn_rna = np.array(train_subset['tpm_unstranded'].tolist()).astype(np.float32)
    trn_dna = np.array(train_subset['beta_value'].tolist()).astype(np.float32)
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn.fit(trn_rna, trn_dna)
    knn_preds = knn.predict(np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32))
    evaluate_reconstruction(true_dna_te, knn_preds, 'KNN', 'DNA')
    
    # 3. Evaluate Conditioned KNN
    trn_sites = train_subset['primary_site_encoded'].values[:, np.newaxis]
    tst_sites = test_subset['primary_site_encoded'].values[:, np.newaxis]
    
    cond_knn = ConditionedKNeighborsRegressor(n_neighbors=5)
    cond_knn.fit(np.hstack((trn_rna, trn_sites)), trn_dna)
    cond_knn_preds = cond_knn.predict(np.hstack((np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32), tst_sites)))
    evaluate_reconstruction(true_dna_te, cond_knn_preds, 'Conditioned KNN', 'DNA')
    
    # Eval VAE (Assuming we can query the trained VAE on the test subset)
    print("\n  Computing VAE reconstruction accuracy on the test subset via subprocess...")
    test_subset.to_pickle('data/temp_test_subset.pkl')
    eval_vae_script = """
import sys
import os
import pandas as pd
import numpy as np
import torch
import json
import glob
from scipy.stats import pearsonr

# We need to load their VAE model
sys.path.append(os.path.abspath('.'))
from src.models import RNA2DNAVAE
from src.config import Config
import pickle

test_subset = pd.read_pickle('data/temp_test_subset.pkl')
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
Config.DEVICE = device

with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

if 'primary_site_encoded' in test_subset.columns:
    site_labels = test_subset['primary_site_encoded'].values
else:
    site_labels = label_encoder.transform(test_subset['primary_site'])

model = RNA2DNAVAE(
    Config.INPUT_DIM_A, 
    Config.INPUT_DIM_B, 
    len(label_encoder.classes_), 
    Config.LATENT_DIM
).to(device)

def get_run_id():
    if os.path.exists('latest_rna2dna_run_id.txt'):
        with open('latest_rna2dna_run_id.txt', 'r') as f:
            return f.read().strip()
    return None

run_id = get_run_id()
model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_rna2dna_{run_id}.pt") if run_id else None

if model_path and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rna_t = torch.tensor(true_rna_te, dtype=torch.float32).to(device)
    site_t = torch.tensor(site_labels).long().to(device)
    
    with torch.no_grad():
        pred_dna, _, _ = model(rna=rna_t, site=site_t)
        pred_dna = pred_dna.cpu().numpy()

    t_flat = true_dna_te.flatten()
    p_flat = pred_dna.flatten()
    mse = float(np.mean((t_flat - p_flat)**2))
    pearson = float(pearsonr(t_flat, p_flat)[0])

    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'MSE': mse, 'Pearson r': pearson}, f)
else:
    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'error': f'No best model {model_path} found'}, f)
"""
    with open('temp_vae_eval.py', 'w') as f:
        f.write(eval_vae_script)
        
    import subprocess
    import json
    try:
        subprocess.run(['venv/bin/python', 'temp_vae_eval.py'], check=True, capture_output=True)
        if os.path.exists('data/temp_vae_acc.json'):
            with open('data/temp_vae_acc.json', 'r') as f:
                vae_results = json.load(f)
            if 'error' not in vae_results:
                print(f"  VAE (DNA): MSE = {vae_results['MSE']:.4f}, Pearson r = {vae_results['Pearson r']:.4f}")
                acc_results.append({
                    'Method': 'VAE',
                    'Target': 'DNA',
                    'MSE': vae_results['MSE'],
                    'Pearson r': vae_results['Pearson r']
                })
            else:
                print(f"  Failed to evaluate VAE accuracy: {vae_results['error']}")
    except Exception as e:
        print(f"  Failed to evaluate VAE accuracy via subprocess: {e}")
    finally:
        if os.path.exists('data/temp_test_subset.pkl'): os.remove('data/temp_test_subset.pkl')
        if os.path.exists('temp_vae_eval.py'): os.remove('temp_vae_eval.py')
        if os.path.exists('data/temp_vae_acc.json'): os.remove('data/temp_vae_acc.json')

    # 5. Evaluate MIMIR (MIMIR was trained on ALL train_df, but we can evaluate on the test_subset for an upper bound info/reference)
    # Since MIMIR relies on complex relative imports natively inside its own directory,
    # we'll write the test_subset to a temp file and execute a subprocess.
    print("\n  Computing MIMIR reconstruction accuracy on the test subset via subprocess...")
    import subprocess
    import json
    
    # Pre-calculate dimensions to avoid module import collisions inside the subprocess
    # Pre-calculate dimensions to avoid module import collisions inside the subprocess
    rna_dim = Config.INPUT_DIM_A
    dna_dim = Config.INPUT_DIM_B

    test_subset.to_pickle('data/temp_test_subset.pkl')
    eval_script = f"""
import sys
import os

import pandas as pd
import numpy as np
import torch
import json
from scipy.stats import pearsonr

# Now we need MIMIR's src.mae_masked, so temporarily restore standard path resolution 
# by prioritizing the local MIMIR directory path
sys.path.insert(0, os.path.abspath('.'))
from src.mae_masked import MultiModalWithSharedSpace, load_modality_with_config, extract_encoder_decoder_from_pretrained

test_subset = pd.read_pickle('../data/temp_test_subset.pkl')
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

encoders, decoders, hidden_dims = {{}}, {{}}, {{}}

# Load pre-trained AEs
rna_ae, rna_hidden, _ = load_modality_with_config('mimir_checkpoints/rna_ae.pt', map_location=device)
dna_ae, dna_hidden, _ = load_modality_with_config('mimir_checkpoints/dna_ae.pt', map_location=device)

encoders['rna'], decoders['rna'] = extract_encoder_decoder_from_pretrained(rna_ae.to(device))
hidden_dims['rna'] = rna_hidden

encoders['dna'], decoders['dna'] = extract_encoder_decoder_from_pretrained(dna_ae.to(device))
hidden_dims['dna'] = dna_hidden

shared_model = MultiModalWithSharedSpace(
    encoders=encoders,
    decoders=decoders,
    hidden_dims=hidden_dims,
    shared_dim=128, proj_depth=1
).to(device)

shared_model.load_state_dict(torch.load('mimir_checkpoints/finetuned/shared_model_ep100.pt', map_location=device, weights_only=True))
shared_model.eval()

rna_val_t = torch.tensor(true_rna_te, dtype=torch.float32).to(device)
res = []
bs = 256
with torch.no_grad():
    for i in range(0, rna_val_t.size(0), bs):
        batch = rna_val_t[i:i+bs]
        encoded = shared_model.encoders['rna'](batch)
        shared = shared_model.projections['rna'](encoded)
        from_shared = shared_model.rev_projections['dna'](shared)
        imputed_dna = shared_model.decoders['dna'](from_shared)
        res.append(imputed_dna.cpu().numpy())
mimir_preds = np.concatenate(res, axis=0)

t_flat = true_dna_te.flatten()
p_flat = mimir_preds.flatten()
mse = float(np.mean((t_flat - p_flat)**2))
pearson = float(pearsonr(t_flat, p_flat)[0])

with open('../data/temp_mimir_acc.json', 'w') as f:
    json.dump({{'MSE': mse, 'Pearson r': pearson}}, f)
"""
    with open('MIMIR/temp_eval.py', 'w') as f:
        f.write(eval_script)
        
    try:
        subprocess.run(['../venv/bin/python', 'temp_eval.py'], cwd='MIMIR', check=True, capture_output=True)
        with open('data/temp_mimir_acc.json', 'r') as f:
            mimir_results = json.load(f)
        
        print(f"  MIMIR (DNA): MSE = {mimir_results['MSE']:.4f}, Pearson r = {mimir_results['Pearson r']:.4f}")
        acc_results.append({
            'Method': 'MIMIR',
            'Target': 'DNA',
            'MSE': mimir_results['MSE'],
            'Pearson r': mimir_results['Pearson r']
        })
    except Exception as e:
        print(f"  Failed to evaluate MIMIR accuracy via subprocess: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  MIMIR STDERR: {e.stderr.decode()}")
    finally:
        if os.path.exists('data/temp_test_subset.pkl'): os.remove('data/temp_test_subset.pkl')
        if os.path.exists('MIMIR/temp_eval.py'): os.remove('MIMIR/temp_eval.py')
        if os.path.exists('data/temp_mimir_acc.json'): os.remove('data/temp_mimir_acc.json')

    # Plot Accuracy Metrics
    if len(acc_results) > 0:
        acc_df = pd.DataFrame(acc_results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = acc_df['Method']
        x = np.arange(len(methods))
        bar_colors = [METHOD_COLORS.get(m, '#cccccc') for m in methods]
        
        axes[0].bar(x, acc_df['MSE'], color=bar_colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Mean Squared Error (Lower is better)')
        axes[0].set_title('Reconstruction MSE (RNA -> DNA)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        
        axes[1].bar(x, acc_df['Pearson r'], color=bar_colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Pearson Correlation r (Higher is better)')
        axes[1].set_title('Reconstruction Correlation (RNA -> DNA)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        acc_filename = f'plots/clustering_mimir/reconstruction_accuracy_{run_timestamp}.png'
        plt.savefig(acc_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy metrics plot saved to: {acc_filename}")
        plt.close()

    print("="*80)
    print("CONTINUING WITH CLUSTERING EVALUATION ON UNMATCHED SAMPLES")
    print("="*80 + "\n")
    rna_only_mean, dna_only_mean = apply_mean_imputation(train_df, rna_only_df, dna_only_df)
    
    # Apply KNN imputation
    rna_only_knn, dna_only_knn = apply_knn_imputation(train_df, rna_only_df, dna_only_df, n_neighbors=5)
    
    # Apply Conditioned KNN imputation
    rna_only_cond_knn, dna_only_cond_knn = apply_conditioned_knn_imputation(train_df, rna_only_df, dna_only_df, label_encoder, n_neighbors=5)
    
    metrics_results = []
    def run_and_store(df, method, sample_type):
        if df is not None and len(df) > 0:
            if sample_type == 'DNA-only' and 'primary_site' not in df.columns:
                print(f"\n⚠ {sample_type} samples don't have primary_site information")
                print(f"  Skipping visualization for {sample_type} samples with {method} imputation")
                return
            outs = analyze_samples(df, label_encoder, run_timestamp, method, sample_type)
            if len(outs) == 4 and outs[3] is not None:
                m = outs[3]
                metrics_results.append({'Method': method, 'Sample Type': sample_type, **m})

    run_and_store(rna_only_mean, "Mean", "RNA-only")
    run_and_store(dna_only_mean, "Mean", "DNA-only")
    run_and_store(rna_only_knn, "KNN", "RNA-only")
    run_and_store(dna_only_knn, "KNN", "DNA-only")
    run_and_store(rna_only_cond_knn, "Conditioned KNN", "RNA-only")
    run_and_store(dna_only_cond_knn, "Conditioned KNN", "DNA-only")
    run_and_store(rna_only_vae, "VAE", "RNA-only")
    run_and_store(dna_only_vae, "VAE", "DNA-only")
    run_and_store(rna_only_mimir, "MIMIR", "RNA-only")
    run_and_store(dna_only_mimir, "MIMIR", "DNA-only")

    # Plot results summary
    if len(metrics_results) > 0:
        plot_metrics_summary(metrics_results, run_timestamp)
    
    print("\n" + "="*80)
    print("Visualization analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
