"""
Script to compare RNA2DNAVAE and DNA2RNAVAE imputation methods
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from src.config import Config
from src.models import RNA2DNAVAE, DNA2RNAVAE
from src.data import MultiModalDataset


def get_run_ids():
    """Get run IDs for both models"""
    rna2dna_run_id = None
    dna2rna_run_id = None
    
    if os.path.exists('latest_rna2dna_run_id.txt'):
        with open('latest_rna2dna_run_id.txt', 'r') as f:
            rna2dna_run_id = f.read().strip()
    
    if os.path.exists('latest_dna2rna_run_id.txt'):
        with open('latest_dna2rna_run_id.txt', 'r') as f:
            dna2rna_run_id = f.read().strip()
    
    return rna2dna_run_id, dna2rna_run_id


def load_models_and_data():
    """Load trained models and validation data"""
    rna2dna_run_id, dna2rna_run_id = get_run_ids()
    
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    n_sites = len(label_encoder.classes_)
    
    # Split data
    train_df, val_df = train_test_split(
        merged_df, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = MultiModalDataset(train_df)
    val_dataset = MultiModalDataset(val_df)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
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
            rna2dna_model.load_state_dict(torch.load(model_path))
            rna2dna_model.eval()
        else:
            print(f"Warning: Model file {model_path} not found!")
            rna2dna_model = None
    else:
        print("Warning: No RNA2DNAVAE run ID found. Please train the model first.")
    
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
            dna2rna_model.load_state_dict(torch.load(model_path))
            dna2rna_model.eval()
        else:
            print(f"Warning: Model file {model_path} not found!")
            dna2rna_model = None
    else:
        print("Warning: No DNA2RNAVAE run ID found. Please train the model first.")
    
    return rna2dna_model, dna2rna_model, val_dataloader, train_dataset, val_dataset, rna2dna_run_id, dna2rna_run_id


def get_predictions(model, dataloader, model_type):
    """Get predictions from a directional VAE model"""
    print(f"\nGenerating predictions from {model_type}...")
    
    rna_true_batches, dna_true_batches = [], []
    rna_pred_batches, dna_pred_batches = [], []
    
    with torch.no_grad():
        for tpm_batch, beta_batch, site_batch in dataloader:
            tpm_batch = tpm_batch.to(Config.DEVICE)
            beta_batch = beta_batch.to(Config.DEVICE)
            site_batch = site_batch.to(Config.DEVICE)
            
            if model_type == 'RNA2DNA':
                # Predict DNA from RNA + site
                recon_dna, _, _ = model(rna=tpm_batch, site=site_batch)
                rna_true_batches.append(tpm_batch.cpu().numpy())
                dna_true_batches.append(beta_batch.cpu().numpy())
                dna_pred_batches.append(recon_dna.cpu().numpy())
                # For RNA, we use the original (no prediction needed)
                rna_pred_batches.append(tpm_batch.cpu().numpy())
                
            elif model_type == 'DNA2RNA':
                # Predict RNA from DNA + site
                recon_rna, _, _ = model(dna=beta_batch, site=site_batch)
                rna_true_batches.append(tpm_batch.cpu().numpy())
                dna_true_batches.append(beta_batch.cpu().numpy())
                rna_pred_batches.append(recon_rna.cpu().numpy())
                # For DNA, we use the original (no prediction needed)
                dna_pred_batches.append(beta_batch.cpu().numpy())
    
    rna_true = np.concatenate(rna_true_batches, axis=0)
    dna_true = np.concatenate(dna_true_batches, axis=0)
    rna_pred = np.concatenate(rna_pred_batches, axis=0)
    dna_pred = np.concatenate(dna_pred_batches, axis=0)
    
    return rna_true, dna_true, rna_pred, dna_pred


def compute_metrics(y_true, y_pred, modality_name, model_name):
    """Compute evaluation metrics"""
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Overall metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Cosine similarity (per sample)
    cos_sim = cosine_similarity(y_true, y_pred)
    cos_sim_mean = float(np.diag(cos_sim).mean())
    
    # Pearson correlation (per sample)
    pearson_all = []
    for i in range(len(y_true)):
        try:
            r, _ = pearsonr(y_true[i], y_pred[i])
            if not np.isnan(r):
                pearson_all.append(r)
        except:
            pass
    
    pearson_mean = np.mean(pearson_all) if pearson_all else 0.0
    pearson_std = np.std(pearson_all) if pearson_all else 0.0
    
    result = {
        "Modality": modality_name,
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2,
        "CosineSimilarity": cos_sim_mean,
        "PearsonMean": pearson_mean,
        "PearsonStd": pearson_std
    }
    
    # Store per-sample correlations for plotting
    result['_pearson_all'] = pearson_all
    
    return result


def get_mean_imputation_predictions(val_dataset):
    """Get mean imputation predictions"""
    print("Computing mean imputation predictions...")
    
    # Fit imputers on the validation data
    rna_imputer = SimpleImputer(strategy="mean")
    dna_imputer = SimpleImputer(strategy="mean")
    
    rna_imputer.fit(val_dataset.tpm_data)
    dna_imputer.fit(val_dataset.beta_data)
    
    # Get mean vectors
    rna_mean_vector = rna_imputer.statistics_.astype(np.float32)
    dna_mean_vector = dna_imputer.statistics_.astype(np.float32)
    
    # Create predictions by repeating mean vectors for all samples
    rna_mean_pred = np.tile(rna_mean_vector, (len(val_dataset), 1))
    dna_mean_pred = np.tile(dna_mean_vector, (len(val_dataset), 1))
    
    return rna_mean_pred, dna_mean_pred


def get_knn_predictions(train_dataset, val_dataset, n_neighbors=5):
    """Get k-NN predictions for cross-modal reconstruction"""
    print(f"Computing k-NN (k={n_neighbors}) predictions...")

    rna_train = train_dataset.tpm_data
    dna_train = train_dataset.beta_data
    rna_val_true = val_dataset.tpm_data
    dna_val_true = val_dataset.beta_data
    
    # Predict RNA from DNA
    knn_rna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    knn_rna.fit(dna_train, rna_train)
    rna_knn_pred = knn_rna.predict(dna_val_true)
    
    # Predict DNA from RNA
    knn_dna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    knn_dna.fit(rna_train, dna_train)
    dna_knn_pred = knn_dna.predict(rna_val_true)
    
    return rna_knn_pred, dna_knn_pred


def plot_comparison(rna_true, dna_true, 
                    rna2dna_dna_pred, dna2rna_rna_pred,
                    dna_mean_pred, rna_mean_pred,
                    dna_knn_pred, rna_knn_pred,
                    output_dir, n_samples=3):
    """Create comparison plots"""
    print(f"\nCreating comparison plots ({n_samples} samples)...")
    
    sample_indices = np.random.choice(len(rna_true), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # DNA prediction comparison
        axes[0, 0].plot(dna_true[idx], label="Actual DNA", alpha=0.8, linewidth=2, color='black')
        axes[0, 0].plot(rna2dna_dna_pred[idx], label="RNA2DNAVAE", alpha=0.8, linewidth=2, color='blue')
        axes[0, 0].plot(dna_mean_pred[idx], label="Mean Imputation", alpha=0.8, linewidth=2, color='green')
        axes[0, 0].plot(dna_knn_pred[idx], label="k-NN Imputation", alpha=0.8, linewidth=2, color='purple')
        axes[0, 0].set_title(f"DNA Methylation Prediction (Sample {i+1})")
        axes[0, 0].set_xlabel("Probe Index")
        axes[0, 0].set_ylabel("Beta Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RNA prediction comparison
        axes[0, 1].plot(rna_true[idx], label="Actual RNA", alpha=0.8, linewidth=2, color='black')
        axes[0, 1].plot(dna2rna_rna_pred[idx], label="DNA2RNAVAE", alpha=0.8, linewidth=2, color='red')
        axes[0, 1].plot(rna_mean_pred[idx], label="Mean Imputation", alpha=0.8, linewidth=2, color='green')
        axes[0, 1].plot(rna_knn_pred[idx], label="k-NN Imputation", alpha=0.8, linewidth=2, color='purple')
        axes[0, 1].set_title(f"RNA Expression Prediction (Sample {i+1})")
        axes[0, 1].set_xlabel("Gene Index")
        axes[0, 1].set_ylabel("Expression Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # DNA scatter plot
        axes[1, 0].scatter(dna_true[idx], rna2dna_dna_pred[idx], alpha=0.6, s=20, color='blue', label="RNA2DNAVAE")
        axes[1, 0].scatter(dna_true[idx], dna_mean_pred[idx], alpha=0.6, s=20, color='green', label="Mean")
        axes[1, 0].scatter(dna_true[idx], dna_knn_pred[idx], alpha=0.6, s=20, color='purple', label="k-NN")
        min_val = min(dna_true[idx].min(), rna2dna_dna_pred[idx].min(), dna_mean_pred[idx].min(), dna_knn_pred[idx].min())
        max_val = max(dna_true[idx].max(), rna2dna_dna_pred[idx].max(), dna_mean_pred[idx].max(), dna_knn_pred[idx].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[1, 0].set_title("DNA: Actual vs Predicted")
        axes[1, 0].set_xlabel("Actual DNA")
        axes[1, 0].set_ylabel("Predicted DNA")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RNA scatter plot
        axes[1, 1].scatter(rna_true[idx], dna2rna_rna_pred[idx], alpha=0.6, s=20, color='red', label="DNA2RNAVAE")
        axes[1, 1].scatter(rna_true[idx], rna_mean_pred[idx], alpha=0.6, s=20, color='green', label="Mean")
        axes[1, 1].scatter(rna_true[idx], rna_knn_pred[idx], alpha=0.6, s=20, color='purple', label="k-NN")
        min_val = min(rna_true[idx].min(), dna2rna_rna_pred[idx].min(), rna_mean_pred[idx].min(), rna_knn_pred[idx].min())
        max_val = max(rna_true[idx].max(), dna2rna_rna_pred[idx].max(), rna_mean_pred[idx].max(), rna_knn_pred[idx].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[1, 1].set_title("RNA: Actual vs Predicted")
        axes[1, 1].set_xlabel("Actual RNA")
        axes[1, 1].set_ylabel("Predicted RNA")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_samples} comparison plots to {output_dir}/")


def plot_correlation_distributions(rna2dna_metrics, dna2rna_metrics, output_dir):
    """Plot Pearson correlation distributions"""
    print("\nGenerating correlation distribution plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # DNA correlation (RNA2DNAVAE)
    if '_pearson_all' in rna2dna_metrics and len(rna2dna_metrics['_pearson_all']) > 0:
        pearson_dna = rna2dna_metrics['_pearson_all']
        axes[0].hist(
            pearson_dna, 
            bins=30, 
            color='skyblue', 
            edgecolor='black', 
            alpha=0.7
        )
        axes[0].axvline(
            rna2dna_metrics['PearsonMean'], 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Mean: {rna2dna_metrics["PearsonMean"]:.3f}'
        )
    axes[0].set_title(f"DNA Prediction Correlation (RNA2DNAVAE)")
    axes[0].set_xlabel("Correlation (r)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RNA correlation (DNA2RNAVAE)
    if '_pearson_all' in dna2rna_metrics and len(dna2rna_metrics['_pearson_all']) > 0:
        pearson_rna = dna2rna_metrics['_pearson_all']
        axes[1].hist(
            pearson_rna, 
            bins=30, 
            color='salmon', 
            edgecolor='black', 
            alpha=0.7
        )
        axes[1].axvline(
            dna2rna_metrics['PearsonMean'], 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Mean: {dna2rna_metrics["PearsonMean"]:.3f}'
        )
    axes[1].set_title(f"RNA Prediction Correlation (DNA2RNAVAE)")
    axes[1].set_xlabel("Correlation (r)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'correlation_distributions.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distribution plots to {filename}")


def create_interactive_plot(rna_true, dna_true,
                           rna2dna_dna_pred, dna2rna_rna_pred,
                           dna_mean_pred, rna_mean_pred,
                           dna_knn_pred, rna_knn_pred,
                           output_dir):
    """Create interactive plotly visualization"""
    print("Creating interactive comparison plot...")
    
    sample_idx = 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("DNA Prediction Comparison", "RNA Prediction Comparison", 
                       "DNA Correlation", "RNA Correlation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # DNA reconstruction
    fig.add_trace(go.Scatter(y=dna_true[sample_idx], name="Actual DNA", 
                            line=dict(color='black', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna2dna_dna_pred[sample_idx], name="RNA2DNAVAE", 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=dna_mean_pred[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=dna_knn_pred[sample_idx], name="k-NN Imputation", 
                            line=dict(color='purple', width=2)), row=1, col=1)
    
    # RNA reconstruction
    fig.add_trace(go.Scatter(y=rna_true[sample_idx], name="Actual RNA", 
                            line=dict(color='black', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna2rna_rna_pred[sample_idx], name="DNA2RNAVAE", 
                            line=dict(color='red', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=rna_mean_pred[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=rna_knn_pred[sample_idx], name="k-NN Imputation", 
                            line=dict(color='purple', width=2)), row=1, col=2)
    
    # DNA correlation
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=rna2dna_dna_pred[sample_idx], 
                            mode='markers', name="RNA2DNAVAE", marker=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_mean_pred[sample_idx], 
                            mode='markers', name="Mean", marker=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_knn_pred[sample_idx], 
                            mode='markers', name="k-NN", marker=dict(color='purple')), row=2, col=1)
    
    # RNA correlation
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=dna2rna_rna_pred[sample_idx], 
                            mode='markers', name="DNA2RNAVAE", marker=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_mean_pred[sample_idx], 
                            mode='markers', name="Mean", marker=dict(color='green')), row=2, col=2)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_knn_pred[sample_idx], 
                            mode='markers', name="k-NN", marker=dict(color='purple')), row=2, col=2)
    
    fig.update_xaxes(title_text="Probe Index", row=1, col=1)
    fig.update_xaxes(title_text="Gene Index", row=1, col=2)
    fig.update_xaxes(title_text="Actual Value", row=2, col=1)
    fig.update_xaxes(title_text="Actual Value", row=2, col=2)
    
    fig.update_yaxes(title_text="Beta Value", row=1, col=1)
    fig.update_yaxes(title_text="Expression Value", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=2)
    
    fig.update_layout(
        width=1200, 
        height=800, 
        hovermode="x unified",
        title="Directional VAE vs Baseline Imputation Comparison"
    )
    
    filename = os.path.join(output_dir, 'interactive_comparison.html')
    fig.write_html(filename)
    print(f"Interactive plot saved to {filename}")


def save_results(results_df, output_dir):
    """Save comparison results"""
    # Remove internal fields before saving
    results_df_clean = results_df.drop(columns=['_pearson_all'], errors='ignore')
    
    # Save as CSV
    csv_filename = os.path.join(output_dir, 'comparison_results.csv')
    results_df_clean.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # Save as JSON (also remove internal fields)
    json_filename = os.path.join(output_dir, 'comparison_results.json')
    results_dict = results_df_clean.to_dict('records')
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results also saved to {json_filename}")


def main():
    """Main comparison pipeline"""
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'plots/directional_comparison_{run_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models and data
    rna2dna_model, dna2rna_model, val_dataloader, train_dataset, val_dataset, rna2dna_run_id, dna2rna_run_id = load_models_and_data()
    
    if rna2dna_model is None and dna2rna_model is None:
        print("Error: No models loaded. Please train both models first.")
        return
    
    # Get baseline predictions
    print("\n" + "="*80)
    print("Computing baseline imputation predictions...")
    print("="*80)
    
    # Mean imputation
    rna_mean_pred, dna_mean_pred = get_mean_imputation_predictions(val_dataset)
    
    # KNN imputation
    rna_knn_pred, dna_knn_pred = get_knn_predictions(train_dataset, val_dataset, n_neighbors=5)
    
    # Get predictions
    results = []
    
    if rna2dna_model:
        rna_true, dna_true, _, rna2dna_dna_pred = get_predictions(
            rna2dna_model, val_dataloader, 'RNA2DNA'
        )
        # Compute metrics for DNA prediction - RNA2DNAVAE
        dna_metrics_vae = compute_metrics(
            dna_true, rna2dna_dna_pred, "DNA methylation", "RNA2DNAVAE"
        )
        results.append(dna_metrics_vae)
        
        # Compute metrics for DNA prediction - Mean
        dna_metrics_mean = compute_metrics(
            dna_true, dna_mean_pred, "DNA methylation", "Mean Imputation"
        )
        results.append(dna_metrics_mean)
        
        # Compute metrics for DNA prediction - KNN
        dna_metrics_knn = compute_metrics(
            dna_true, dna_knn_pred, "DNA methylation", "k-NN Imputation"
        )
        results.append(dna_metrics_knn)
        
        print(f"\nDNA Prediction Results:")
        print(f"  RNA2DNAVAE - MAE: {dna_metrics_vae['MAE']:.4f}, MSE: {dna_metrics_vae['MSE']:.4f}, R2: {dna_metrics_vae['R2']:.4f}, Pearson r: {dna_metrics_vae['PearsonMean']:.4f}")
        print(f"  Mean Imputation - MAE: {dna_metrics_mean['MAE']:.4f}, MSE: {dna_metrics_mean['MSE']:.4f}, R2: {dna_metrics_mean['R2']:.4f}, Pearson r: {dna_metrics_mean['PearsonMean']:.4f}")
        print(f"  k-NN Imputation - MAE: {dna_metrics_knn['MAE']:.4f}, MSE: {dna_metrics_knn['MSE']:.4f}, R2: {dna_metrics_knn['R2']:.4f}, Pearson r: {dna_metrics_knn['PearsonMean']:.4f}")
    else:
        rna_true, dna_true = None, None
        rna2dna_dna_pred = None
        dna_metrics_vae = None
    
    if dna2rna_model:
        if rna_true is None:
            rna_true, dna_true, _, _ = get_predictions(
                dna2rna_model, val_dataloader, 'DNA2RNA'
            )
        _, _, dna2rna_rna_pred, _ = get_predictions(
            dna2rna_model, val_dataloader, 'DNA2RNA'
        )
        # Compute metrics for RNA prediction - DNA2RNAVAE
        rna_metrics_vae = compute_metrics(
            rna_true, dna2rna_rna_pred, "RNA expression", "DNA2RNAVAE"
        )
        results.append(rna_metrics_vae)
        
        # Compute metrics for RNA prediction - Mean
        rna_metrics_mean = compute_metrics(
            rna_true, rna_mean_pred, "RNA expression", "Mean Imputation"
        )
        results.append(rna_metrics_mean)
        
        # Compute metrics for RNA prediction - KNN
        rna_metrics_knn = compute_metrics(
            rna_true, rna_knn_pred, "RNA expression", "k-NN Imputation"
        )
        results.append(rna_metrics_knn)
        
        print(f"\nRNA Prediction Results:")
        print(f"  DNA2RNAVAE - MAE: {rna_metrics_vae['MAE']:.4f}, MSE: {rna_metrics_vae['MSE']:.4f}, R2: {rna_metrics_vae['R2']:.4f}, Pearson r: {rna_metrics_vae['PearsonMean']:.4f}")
        print(f"  Mean Imputation - MAE: {rna_metrics_mean['MAE']:.4f}, MSE: {rna_metrics_mean['MSE']:.4f}, R2: {rna_metrics_mean['R2']:.4f}, Pearson r: {rna_metrics_mean['PearsonMean']:.4f}")
        print(f"  k-NN Imputation - MAE: {rna_metrics_knn['MAE']:.4f}, MSE: {rna_metrics_knn['MSE']:.4f}, R2: {rna_metrics_knn['R2']:.4f}, Pearson r: {rna_metrics_knn['PearsonMean']:.4f}")
    else:
        dna2rna_rna_pred = None
        rna_metrics_vae = None
    
    if rna_true is None or dna_true is None:
        print("Error: Could not load validation data.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("DIRECTIONAL VAE IMPUTATION COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    if rna2dna_model and dna2rna_model:
        plot_comparison(
            rna_true, dna_true,
            rna2dna_dna_pred, dna2rna_rna_pred,
            dna_mean_pred, rna_mean_pred,
            dna_knn_pred, rna_knn_pred,
            output_dir, n_samples=3
        )
        
        if dna_metrics_vae and rna_metrics_vae:
            plot_correlation_distributions(dna_metrics_vae, rna_metrics_vae, output_dir)
        
        create_interactive_plot(
            rna_true, dna_true,
            rna2dna_dna_pred, dna2rna_rna_pred,
            dna_mean_pred, rna_mean_pred,
            dna_knn_pred, rna_knn_pred,
            output_dir
        )
    
    # Save results
    save_results(results_df, output_dir)
    
    print(f"\nComparison complete! All results saved to {output_dir}/")


if __name__ == "__main__":
    main()

