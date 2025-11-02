"""
Script to compare trained VAE model with mean and k-NN imputation baselines
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    return model, val_dataloader, val_dataset, run_id


def compute_metrics(modality: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics for a given modality and model"""
    similarities = cosine_similarity(y_true, y_pred)
    return {
        "Modality": modality,
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CosineSimilarity": float(np.diag(similarities).mean())
    }


def get_vae_predictions(model, val_dataloader):
    """Get VAE predictions for cross-modal reconstruction"""
    print("Generating VAE predictions...")
    
    rna_true_batches, rna_vae_batches = [], []
    dna_true_batches, dna_vae_batches = [], []
    
    with torch.no_grad():
        for tpm_batch, beta_batch, site_batch in val_dataloader:
            tpm_batch = tpm_batch.to(Config.DEVICE)
            beta_batch = beta_batch.to(Config.DEVICE)
            site_batch = site_batch.to(Config.DEVICE)
            
            # Cross-modal reconstructions
            recon_rna_from_beta, _, _, _, _ = model(a=None, b=beta_batch, site=site_batch)
            _, recon_beta_from_rna, _, _, _ = model(a=tpm_batch, b=None, site=site_batch)
            
            rna_true_batches.append(tpm_batch.cpu().numpy())
            dna_true_batches.append(beta_batch.cpu().numpy())
            rna_vae_batches.append(recon_rna_from_beta.cpu().numpy())
            dna_vae_batches.append(recon_beta_from_rna.cpu().numpy())
    
    rna_true = np.concatenate(rna_true_batches, axis=0)
    dna_true = np.concatenate(dna_true_batches, axis=0)
    rna_vae_pred = np.concatenate(rna_vae_batches, axis=0)
    dna_vae_pred = np.concatenate(dna_vae_batches, axis=0)
    
    return rna_true, dna_true, rna_vae_pred, dna_vae_pred


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


def get_knn_predictions(rna_true, dna_true, n_neighbors=5):
    """Get k-NN predictions for cross-modal reconstruction"""
    print(f"Computing k-NN (k={n_neighbors}) predictions...")
    
    # Predict RNA from DNA
    knn_rna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    knn_rna.fit(dna_true, rna_true)
    rna_knn_pred = knn_rna.predict(dna_true)
    
    # Predict DNA from RNA
    knn_dna = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    knn_dna.fit(rna_true, dna_true)
    dna_knn_pred = knn_dna.predict(rna_true)
    
    return rna_knn_pred, dna_knn_pred

def create_comparison_plot(rna_true, rna_vae, rna_mean, rna_knn, dna_true, dna_vae, dna_mean, dna_knn, output_dir):
    """Create comparison plot using matplotlib"""
    print("Creating comparison plots...")
    
    # Sample a few examples for visualization
    n_samples = min(3, len(rna_true))
    sample_indices = np.random.choice(len(rna_true), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RNA comparison
        axes[0, 0].plot(rna_true[idx], label="Actual RNA", alpha=0.8, linewidth=2)
        axes[0, 0].plot(rna_vae[idx], label="VAE Prediction", alpha=0.8, linewidth=2)
        axes[0, 0].plot(rna_mean[idx], label="Mean Imputation", alpha=0.8, linewidth=2)
        axes[0, 0].plot(rna_knn[idx], label="k-NN Prediction", alpha=0.8, linewidth=2)
        axes[0, 0].set_title(f"RNA Reconstruction (Sample {i+1})")
        axes[0, 0].set_xlabel("Gene Index")
        axes[0, 0].set_ylabel("Expression Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # DNA comparison
        axes[0, 1].plot(dna_true[idx], label="Actual DNA", alpha=0.8, linewidth=2)
        axes[0, 1].plot(dna_vae[idx], label="VAE Prediction", alpha=0.8, linewidth=2)
        axes[0, 1].plot(dna_mean[idx], label="Mean Imputation", alpha=0.8, linewidth=2)
        axes[0, 1].plot(dna_knn[idx], label="k-NN Prediction", alpha=0.8, linewidth=2)
        axes[0, 1].set_title(f"DNA Methylation Reconstruction (Sample {i+1})")
        axes[0, 1].set_xlabel("Probe Index")
        axes[0, 1].set_ylabel("Beta Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plots for correlation
        axes[1, 0].scatter(rna_true[idx], rna_vae[idx], alpha=0.6, label="VAE", s=20)
        axes[1, 0].scatter(rna_true[idx], rna_mean[idx], alpha=0.6, label="Mean", s=20)
        axes[1, 0].scatter(rna_true[idx], rna_knn[idx], alpha=0.6, label="k-NN", s=20)
        axes[1, 0].plot([rna_true[idx].min(), rna_true[idx].max()], 
                       [rna_true[idx].min(), rna_true[idx].max()], 'k--', alpha=0.5)
        axes[1, 0].set_title("RNA: Actual vs Predicted")
        axes[1, 0].set_xlabel("Actual RNA")
        axes[1, 0].set_ylabel("Predicted")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(dna_true[idx], dna_vae[idx], alpha=0.6, label="VAE", s=20)
        axes[1, 1].scatter(dna_true[idx], dna_mean[idx], alpha=0.6, label="Mean", s=20)
        axes[1, 1].scatter(dna_true[idx], dna_knn[idx], alpha=0.6, label="k-NN", s=20)
        axes[1, 1].plot([dna_true[idx].min(), dna_true[idx].max()], 
                       [dna_true[idx].min(), dna_true[idx].max()], 'k--', alpha=0.5)
        axes[1, 1].set_title("DNA: Actual vs Predicted")
        axes[1, 1].set_xlabel("Actual DNA")
        axes[1, 1].set_ylabel("Predicted DNA")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_samples} comparison plots to {output_dir}/")

def create_interactive_plot(rna_true, rna_vae, rna_mean, rna_knn, dna_true, dna_vae, dna_mean, dna_knn):
    """Create interactive plotly visualization"""
    print("Creating interactive comparison plot...")
    
    # Sample one example for interactive plot
    sample_idx = 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("RNA Reconstruction", "DNA Methylation Reconstruction", 
                       "RNA Correlation", "DNA Correlation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RNA reconstruction
    fig.add_trace(go.Scatter(y=rna_true[sample_idx], name="Actual RNA", 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_vae[sample_idx], name="VAE Prediction", 
                            line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_mean[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_knn[sample_idx], name="k-NN Prediction", 
                            line=dict(color='purple', width=2)), row=1, col=1)
    
    # DNA reconstruction
    fig.add_trace(go.Scatter(y=dna_true[sample_idx], name="Actual DNA", 
                            line=dict(color='blue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_vae[sample_idx], name="VAE Prediction", 
                            line=dict(color='red', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_mean[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_knn[sample_idx], name="k-NN Prediction", 
                            line=dict(color='purple', width=2)), row=1, col=2)
    
    # RNA correlation
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_vae[sample_idx], 
                            mode='markers', name="VAE", marker=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_mean[sample_idx], 
                            mode='markers', name="Mean", marker=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_knn[sample_idx], 
                            mode='markers', name="k-NN", marker=dict(color='purple')), row=2, col=1)
    
    # DNA correlation
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_vae[sample_idx], 
                            mode='markers', name="VAE", marker=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_mean[sample_idx], 
                            mode='markers', name="Mean", marker=dict(color='green')), row=2, col=2)
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_knn[sample_idx], 
                            mode='markers', name="k-NN", marker=dict(color='purple')), row=2, col=2)
    
    fig.update_xaxes(title_text="Feature Index", row=1, col=1)
    fig.update_xaxes(title_text="Probe Index", row=1, col=2)
    fig.update_xaxes(title_text="Actual Value", row=2, col=1)
    fig.update_xaxes(title_text="Actual Value", row=2, col=2)
    
    fig.update_yaxes(title_text="Expression Value", row=1, col=1)
    fig.update_yaxes(title_text="Beta Value", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=2)
    
    fig.update_layout(
        width=1200, 
        height=800, 
        hovermode="x unified",
        title="VAE vs Mean vs k-NN Imputation Comparison"
    )
    
    return fig

def save_results(results_df, output_dir):
    """Save comparison results"""
    # Save as CSV
    csv_filename = os.path.join(output_dir, 'comparison_results.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # Save as JSON
    import json
    json_filename = os.path.join(output_dir, 'comparison_results.json')
    results_dict = results_df.to_dict('records')
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results also saved to {json_filename}")

def main():
    """Main comparison pipeline"""
    # Create plots directory
    run_id = get_run_id()
    run_suffix = f"_{run_id}" if run_id else f"_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = f'plots/comparison{run_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model, val_dataloader, val_dataset, run_id = load_model_and_data()
    
    # Get VAE predictions
    rna_true, dna_true, rna_vae_pred, dna_vae_pred = get_vae_predictions(model, val_dataloader)
    
    # Get mean imputation predictions
    rna_mean_pred, dna_mean_pred = get_mean_imputation_predictions(val_dataset)
    
    # Get k-NN predictions
    rna_knn_pred, dna_knn_pred = get_knn_predictions(rna_true, dna_true)
    
    print(f"Total validation samples: {len(rna_true)}")
    print(f"RNA dimension: {rna_true.shape[1]}")
    print(f"DNA dimension: {dna_true.shape[1]}")
    
    # Compute comparison metrics
    results = [
        compute_metrics("RNA", "VAE", rna_true, rna_vae_pred),
        compute_metrics("RNA", "Mean Imputation", rna_true, rna_mean_pred),
        compute_metrics("RNA", "k-NN Imputation", rna_true, rna_knn_pred),
        compute_metrics("DNA methylation", "VAE", dna_true, dna_vae_pred),
        compute_metrics("DNA methylation", "Mean Imputation", dna_true, dna_mean_pred),
        compute_metrics("DNA methylation", "k-NN Imputation", dna_true, dna_knn_pred)
    ]
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARISON RESULTS: VAE vs MEAN vs k-NN IMPUTATION")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    create_comparison_plot(rna_true, rna_vae_pred, rna_mean_pred, rna_knn_pred,
                          dna_true, dna_vae_pred, dna_mean_pred, dna_knn_pred, output_dir)
    
    # Create interactive plot
    interactive_fig = create_interactive_plot(rna_true, rna_vae_pred, rna_mean_pred, rna_knn_pred,
                                            dna_true, dna_vae_pred, dna_mean_pred, dna_knn_pred)
    
    interactive_fig.write_html(os.path.join(output_dir, 'interactive_comparison.html'))
    print(f"Interactive plot saved to {os.path.join(output_dir, 'interactive_comparison.html')}")
    
    # Save results
    save_results(results_df, output_dir)
    
    print(f"\nComparison complete! All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
