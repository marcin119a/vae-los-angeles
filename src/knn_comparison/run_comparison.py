import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import Config
from src.models import DNA2RNAVAE, RNA2DNAVAE
from src.data import MultiModalDataset
from src.knn_comparison.conditioned_knn import ConditionedKNeighborsRegressor

# Set plotly template
pio.templates.default = "plotly_white"

def load_processed_data():
    """Load processed data and label encoder"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
        
    return merged_df, label_encoder

def prepare_arrays(df, label_encoder):
    """Convert dataframe columns to numpy arrays and one-hot encode sites"""
    print("Preparing data arrays...")
    
    # Extract features
    X_rna = np.stack(df['tpm_unstranded'].values).astype(np.float32)
    X_dna = np.stack(df['beta_value'].values).astype(np.float32)
    y_site_idx = df['primary_site_encoded'].values.reshape(-1, 1)
    
    # One-hot encode sites
    encoder = OneHotEncoder(sparse_output=False)
    X_site = encoder.fit_transform(y_site_idx).astype(np.float32)
    
    return X_rna, X_dna, X_site, y_site_idx.flatten()

def optimize_knn(X_train, y_train, X_val, y_val, name="KNN", model_class=KNeighborsRegressor):
    """Optimize KNN hyperparameters using GridSearchCV"""
    print(f"\nOptimizing {name}...")
    
    # We use a subset for grid search if the dataset is huge, but here it's likely manageable
    # However, standard KNN is slow on inference.
    
    params = {
        'n_neighbors': [5, 10, 20, 50],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # We can use the validation set explicitly or cross-validation. 
    # To save time and match the VAE split structure, we will just train on Train and evaluate on Val
    # for a manual grid search loop instead of CV to be faster and simpler.
    
    best_mse = float('inf')
    best_model = None
    best_params = {}
    
    # Iterate over parameters
    import itertools
    keys, values = zip(*params.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for p in tqdm(param_combinations, desc="Grid Search"):
        model = model_class(**p)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = p
            
    print(f"Best {name} params: {best_params} | MSE: {best_mse:.4f}")
    return best_model

def load_vae_model(model_class, run_id_file, input_dim_a, input_dim_b, n_sites):
    """Load a trained VAE model"""
    with open(run_id_file, 'r') as f:
        run_id = f.read().strip()
        
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'best_{model_class.__name__.lower().replace("vae", "")}_{run_id}.pt')
    
    print(f"Loading {model_class.__name__} from {checkpoint_path}...")
    
    model = model_class(
        input_dim_a, 
        input_dim_b, 
        n_sites, 
        Config.LATENT_DIM
    ).to(Config.DEVICE)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    return model

def get_vae_predictions(model, loader, direction='dna2rna'):
    """Get predictions from VAE"""
    preds = []
    targets = []
    
    with torch.no_grad():
        for tpm, beta, site in tqdm(loader, desc=f"Eval VAE {direction}"):
            tpm = tpm.to(Config.DEVICE)
            beta = beta.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            
            if direction == 'dna2rna':
                recon, _, _ = model(dna=beta, site=site)
                preds.append(recon.cpu().numpy())
                targets.append(tpm.cpu().numpy())
            else: # rna2dna
                recon, _, _ = model(rna=tpm, site=site)
                preds.append(recon.cpu().numpy())
                targets.append(beta.cpu().numpy())
                
    return np.vstack(preds), np.vstack(targets)

def create_boxplots(results, direction):
    """Create boxplots for comparison"""
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for model_name, mses in results.items():
        plot_data.append(mses)
        labels.append(f"{model_name}\n(Mean: {np.mean(mses):.4f})")
        
    # Matplotlib
    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, tick_labels=labels, patch_artist=True)
    plt.title(f'Reconstruction Error Distribution ({direction})')
    plt.ylabel('Mean Squared Error (per sample)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    
    os.makedirs('plots/comparison', exist_ok=True)
    plt_path = f'plots/comparison/boxplot_{direction}.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved matplotlib plot to {plt_path}")
    
    # Plotly
    fig = go.Figure()
    for name, mses in results.items():
        fig.add_trace(go.Box(y=mses, name=name, boxpoints='outliers')) # show only outliers to keep it clean or 'all'
        
    fig.update_layout(
        title=f'Reconstruction Error Distribution ({direction})',
        yaxis_title='Mean Squared Error',
        xaxis_title='Model',
        template='plotly_white'
    )
    
    plotly_path = f'plots/comparison/boxplot_{direction}.html'
    fig.write_html(plotly_path)
    print(f"Saved plotly plot to {plotly_path}")


def compute_and_plot_tsne(data, site_labels, title, filename_prefix):
    """Compute t-SNE and generate plots (Matplotlib/Seaborn + Plotly)"""
    print(f"Computing t-SNE for {title}...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
    tsne_results = tsne.fit_transform(data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'Primary Site': site_labels
    })
    
    # 1. Matplotlib + Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x='tsne_1',
        y='tsne_2',
        hue='Primary Site',
        palette='tab10',
        s=100,
        alpha=0.7
    )
    plt.title(f't-SNE: {title}')
    plt.grid(True, alpha=0.3)
    
    # Move legend outside if too many classes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt_path = f'plots/comparison/tsne_{filename_prefix}.png'
    os.makedirs(os.path.dirname(plt_path), exist_ok=True)
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {plt_path}")
    
    # 2. Plotly
    fig = px.scatter(
        plot_df, 
        x='tsne_1', 
        y='tsne_2', 
        color='Primary Site',
        title=f't-SNE: {title}',
        template='plotly_white',
        hover_data=['Primary Site']
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    
    plotly_path = f'plots/comparison/tsne_{filename_prefix}.html'
    fig.write_html(plotly_path)
    print(f"Saved plotly t-SNE to {plotly_path}")


def main():
    # 1. Load Data
    df, label_encoder = load_processed_data()
    X_rna, X_dna, X_site, y_site = prepare_arrays(df, label_encoder)
    n_sites = len(label_encoder.classes_)
    
    # 2. Split Data (using same seed as VAE training)
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    X_rna_train, X_rna_val = X_rna[train_idx], X_rna[val_idx]
    X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
    X_site_train, X_site_val = X_site[train_idx], X_site[val_idx]
    y_site_train, y_site_val = y_site[train_idx], y_site[val_idx]
    y_site_val_labels = label_encoder.inverse_transform(y_site_val)
    
    # --- Part 1: RNA -> DNA ---
    print("\n" + "="*50)
    print("COMPARISON: RNA -> DNA")
    print("="*50)
    
    results_rna2dna = {}
    
    # A. KNN (RNA -> DNA)
    knn_rna2dna = optimize_knn(X_rna_train, X_dna_train, X_rna_val, X_dna_val, name="KNN (Base)")
    preds_base = knn_rna2dna.predict(X_rna_val)
    mse_per_sample = np.mean((preds_base - X_dna_val)**2, axis=1)
    results_rna2dna["KNN (Base)"] = mse_per_sample
    compute_and_plot_tsne(preds_base, y_site_val_labels, "KNN (Base) RNA->DNA", "rna2dna_knn_base")
    
    # B. KNN (RNA + Site -> DNA)
    # New strategy: Use Stratified/Conditioned KNN
    # Prepare data: Append site index as last column
    X_train_cond = np.column_stack([X_rna_train, y_site_train])
    X_val_cond = np.column_stack([X_rna_val, y_site_val])
    
    knn_rna2dna_cond = optimize_knn(
        X_train_cond, X_dna_train, 
        X_val_cond, X_dna_val, 
        name="KNN (Conditioned)",
        model_class=ConditionedKNeighborsRegressor
    )
    preds_cond = knn_rna2dna_cond.predict(X_val_cond)
    mse_per_sample = np.mean((preds_cond - X_dna_val)**2, axis=1)
    results_rna2dna["KNN (Cond)"] = mse_per_sample
    compute_and_plot_tsne(preds_cond, y_site_val_labels, "KNN (Cond) RNA->DNA", "rna2dna_knn_cond")
    
    # C. VAE (RNA -> DNA)
    vae_model = load_vae_model(RNA2DNAVAE, 'latest_rna2dna_run_id.txt', 
                              Config.INPUT_DIM_A, Config.INPUT_DIM_B, n_sites)
    
    # Create dataloader for VAE eval
    val_df = df.iloc[val_idx]
    val_dataset = MultiModalDataset(val_df)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    vae_preds, vae_targets = get_vae_predictions(vae_model, val_loader, direction='rna2dna')
    mse_per_sample = np.mean((vae_preds - vae_targets)**2, axis=1)
    results_rna2dna["VAE (Cond)"] = mse_per_sample
    compute_and_plot_tsne(vae_preds, y_site_val_labels, "VAE (Cond) RNA->DNA", "rna2dna_vae_cond")
    
    # Create plots
    create_boxplots(results_rna2dna, "RNA_to_DNA")
    
    
    # --- Part 2: DNA -> RNA ---
    print("\n" + "="*50)
    print("COMPARISON: DNA -> RNA")
    print("="*50)
    
    results_dna2rna = {}
    
    # A. KNN (DNA -> RNA)
    knn_dna2rna = optimize_knn(X_dna_train, X_rna_train, X_dna_val, X_rna_val, name="KNN (Base)")
    preds_base = knn_dna2rna.predict(X_dna_val)
    mse_per_sample = np.mean((preds_base - X_rna_val)**2, axis=1)
    results_dna2rna["KNN (Base)"] = mse_per_sample
    compute_and_plot_tsne(preds_base, y_site_val_labels, "KNN (Base) DNA->RNA", "dna2rna_knn_base")
    
    # B. KNN (DNA + Site -> RNA)
    # Append site index
    X_train_cond = np.column_stack([X_dna_train, y_site_train])
    X_val_cond = np.column_stack([X_dna_val, y_site_val])
    
    knn_dna2rna_cond = optimize_knn(
        X_train_cond, X_rna_train, 
        X_val_cond, X_rna_val, 
        name="KNN (Conditioned)",
        model_class=ConditionedKNeighborsRegressor
    )
    preds_cond = knn_dna2rna_cond.predict(X_val_cond)
    mse_per_sample = np.mean((preds_cond - X_rna_val)**2, axis=1)
    results_dna2rna["KNN (Cond)"] = mse_per_sample
    compute_and_plot_tsne(preds_cond, y_site_val_labels, "KNN (Cond) DNA->RNA", "dna2rna_knn_cond")
    
    # C. VAE (DNA -> RNA)
    vae_model = load_vae_model(DNA2RNAVAE, 'latest_dna2rna_run_id.txt', 
                              Config.INPUT_DIM_A, Config.INPUT_DIM_B, n_sites)
    
    # Reuse loader
    vae_preds, vae_targets = get_vae_predictions(vae_model, val_loader, direction='dna2rna')
    mse_per_sample = np.mean((vae_preds - vae_targets)**2, axis=1)
    results_dna2rna["VAE (Cond)"] = mse_per_sample
    compute_and_plot_tsne(vae_preds, y_site_val_labels, "VAE (Cond) DNA->RNA", "dna2rna_vae_cond")

    
    # Create plots
    create_boxplots(results_dna2rna, "DNA_to_RNA")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
