"""
Script to compare trained CVAE model with mean and k-NN imputation baselines
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
from src.models.cvae import ConditionalMultiModalVAE
from src.data import MultiModalDataset


def load_model_and_data():
    """Load trained CVAE model and data"""
    print("Loading processed data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    n_sites = len(label_encoder.classes_)
    
    # Split data (same split as training)
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
    
    # Get dimensions from first batch
    sample_batch = next(iter(val_dataloader))
    input_dim_rna = sample_batch[0].shape[1]
    input_dim_dna = sample_batch[1].shape[1]
    
    # Load CVAE model
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
    
    print(f"CVAE model loaded from epoch {checkpoint['epoch'] + 1}")
    
    return model, val_dataloader, train_dataset, val_dataset


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


def get_cvae_predictions(model, val_dataloader):
    """Get CVAE predictions for cross-modal reconstruction"""
    print("Generating CVAE predictions...")
    
    rna_true_batches, rna_cvae_batches = [], []
    dna_true_batches, dna_cvae_batches = [], []
    
    with torch.no_grad():
        for tpm_batch, beta_batch, site_batch in val_dataloader:
            tpm_batch = tpm_batch.to(Config.DEVICE)
            beta_batch = beta_batch.to(Config.DEVICE)
            site_batch = site_batch.to(Config.DEVICE)
            
            # Cross-modal reconstructions (CVAE requires site labels)
            recon_rna_from_beta, _, _, _, _ = model(a=None, b=beta_batch, site=site_batch)
            _, recon_beta_from_rna, _, _, _ = model(a=tpm_batch, b=None, site=site_batch)
            
            rna_true_batches.append(tpm_batch.cpu().numpy())
            dna_true_batches.append(beta_batch.cpu().numpy())
            rna_cvae_batches.append(recon_rna_from_beta.cpu().numpy())
            dna_cvae_batches.append(recon_beta_from_rna.cpu().numpy())
    
    rna_true = np.concatenate(rna_true_batches, axis=0)
    dna_true = np.concatenate(dna_true_batches, axis=0)
    rna_cvae_pred = np.concatenate(rna_cvae_batches, axis=0)
    dna_cvae_pred = np.concatenate(dna_cvae_batches, axis=0)
    
    return rna_true, dna_true, rna_cvae_pred, dna_cvae_pred


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


def create_comparison_plot(rna_true, rna_cvae, rna_mean, rna_knn, dna_true, dna_cvae, dna_mean, dna_knn, output_dir):
    """Create comparison plot using matplotlib"""
    print("Creating comparison plots...")
    
    # Sample a few examples for visualization
    n_samples = min(3, len(rna_true))
    sample_indices = np.random.choice(len(rna_true), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RNA comparison
        axes[0, 0].plot(rna_true[idx], label="Actual RNA", alpha=0.8, linewidth=2, color='blue')
        axes[0, 0].plot(rna_cvae[idx], label="CVAE Prediction", alpha=0.8, linewidth=2, color='red')
        axes[0, 0].plot(rna_mean[idx], label="Mean Imputation", alpha=0.8, linewidth=2, color='green')
        axes[0, 0].plot(rna_knn[idx], label="k-NN Prediction", alpha=0.8, linewidth=2, color='purple')
        axes[0, 0].set_title(f"RNA Reconstruction (Sample {i+1}) - CVAE vs Baselines")
        axes[0, 0].set_xlabel("Gene Index")
        axes[0, 0].set_ylabel("Expression Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # DNA comparison
        axes[0, 1].plot(dna_true[idx], label="Actual DNA", alpha=0.8, linewidth=2, color='blue')
        axes[0, 1].plot(dna_cvae[idx], label="CVAE Prediction", alpha=0.8, linewidth=2, color='red')
        axes[0, 1].plot(dna_mean[idx], label="Mean Imputation", alpha=0.8, linewidth=2, color='green')
        axes[0, 1].plot(dna_knn[idx], label="k-NN Prediction", alpha=0.8, linewidth=2, color='purple')
        axes[0, 1].set_title(f"DNA Methylation Reconstruction (Sample {i+1}) - CVAE vs Baselines")
        axes[0, 1].set_xlabel("Probe Index")
        axes[0, 1].set_ylabel("Beta Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plots for correlation
        axes[1, 0].scatter(rna_true[idx], rna_cvae[idx], alpha=0.6, label="CVAE", s=20, color='red')
        axes[1, 0].scatter(rna_true[idx], rna_mean[idx], alpha=0.6, label="Mean", s=20, color='green')
        axes[1, 0].scatter(rna_true[idx], rna_knn[idx], alpha=0.6, label="k-NN", s=20, color='purple')
        axes[1, 0].plot([rna_true[idx].min(), rna_true[idx].max()], 
                       [rna_true[idx].min(), rna_true[idx].max()], 'k--', alpha=0.5)
        axes[1, 0].set_title("RNA: Actual vs Predicted")
        axes[1, 0].set_xlabel("Actual RNA")
        axes[1, 0].set_ylabel("Predicted")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(dna_true[idx], dna_cvae[idx], alpha=0.6, label="CVAE", s=20, color='red')
        axes[1, 1].scatter(dna_true[idx], dna_mean[idx], alpha=0.6, label="Mean", s=20, color='green')
        axes[1, 1].scatter(dna_true[idx], dna_knn[idx], alpha=0.6, label="k-NN", s=20, color='purple')
        axes[1, 1].plot([dna_true[idx].min(), dna_true[idx].max()], 
                       [dna_true[idx].min(), dna_true[idx].max()], 'k--', alpha=0.5)
        axes[1, 1].set_title("DNA: Actual vs Predicted")
        axes[1, 1].set_xlabel("Actual DNA")
        axes[1, 1].set_ylabel("Predicted DNA")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cvae_comparison_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_samples} comparison plots to {output_dir}/")


def create_interactive_plot(rna_true, rna_cvae, rna_mean, rna_knn, dna_true, dna_cvae, dna_mean, dna_knn):
    """Create interactive plotly visualization"""
    print("Creating interactive comparison plot...")
    
    # Sample one example for interactive plot
    sample_idx = 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("RNA Reconstruction (CVAE)", "DNA Methylation Reconstruction (CVAE)", 
                       "RNA Correlation", "DNA Correlation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RNA reconstruction
    fig.add_trace(go.Scatter(y=rna_true[sample_idx], name="Actual RNA", 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_cvae[sample_idx], name="CVAE Prediction", 
                            line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_mean[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=rna_knn[sample_idx], name="k-NN Prediction", 
                            line=dict(color='purple', width=2)), row=1, col=1)
    
    # DNA reconstruction
    fig.add_trace(go.Scatter(y=dna_true[sample_idx], name="Actual DNA", 
                            line=dict(color='blue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_cvae[sample_idx], name="CVAE Prediction", 
                            line=dict(color='red', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_mean[sample_idx], name="Mean Imputation", 
                            line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=dna_knn[sample_idx], name="k-NN Prediction", 
                            line=dict(color='purple', width=2)), row=1, col=2)
    
    # RNA correlation
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_cvae[sample_idx], 
                            mode='markers', name="CVAE", marker=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_mean[sample_idx], 
                            mode='markers', name="Mean", marker=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=rna_true[sample_idx], y=rna_knn[sample_idx], 
                            mode='markers', name="k-NN", marker=dict(color='purple')), row=2, col=1)
    
    # DNA correlation
    fig.add_trace(go.Scatter(x=dna_true[sample_idx], y=dna_cvae[sample_idx], 
                            mode='markers', name="CVAE", marker=dict(color='red')), row=2, col=2)
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
        title="CVAE vs Mean vs k-NN Imputation Comparison"
    )
    
    return fig


def create_metrics_bar_chart(results_df, output_dir):
    """Create bar chart comparing metrics across methods"""
    print("Creating metrics comparison bar chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['MAE', 'MSE', 'R2', 'CosineSimilarity']
    titles = ['Mean Absolute Error (lower is better)', 
              'Mean Squared Error (lower is better)',
              'R² Score (higher is better)',
              'Cosine Similarity (higher is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Separate RNA and DNA
        rna_data = results_df[results_df['Modality'] == 'RNA']
        dna_data = results_df[results_df['Modality'] == 'DNA methylation']
        
        x = np.arange(len(rna_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rna_data[metric], width, label='RNA', alpha=0.8)
        bars2 = ax.bar(x + width/2, dna_data[metric], width, label='DNA', alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(rna_data['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cvae_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics comparison bar chart to {output_dir}/")


def save_results(results_df, output_dir):
    """Save comparison results"""
    # Save as CSV
    csv_filename = os.path.join(output_dir, 'cvae_comparison_results.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # Save as JSON
    import json
    json_filename = os.path.join(output_dir, 'cvae_comparison_results.json')
    results_dict = results_df.to_dict('records')
    with open(json_filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results also saved to {json_filename}")


def main():
    """Main comparison pipeline for CVAE"""
    # Create plots directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = f'plots/cvae_comparison_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model, val_dataloader, train_dataset, val_dataset = load_model_and_data()
    
    # Get CVAE predictions
    rna_true, dna_true, rna_cvae_pred, dna_cvae_pred = get_cvae_predictions(model, val_dataloader)
    
    # Get mean imputation predictions
    rna_mean_pred, dna_mean_pred = get_mean_imputation_predictions(val_dataset)
    
    # Get k-NN predictions
    rna_knn_pred, dna_knn_pred = get_knn_predictions(train_dataset, val_dataset)
    
    print(f"\nTotal validation samples: {len(rna_true)}")
    print(f"RNA dimension: {rna_true.shape[1]}")
    print(f"DNA dimension: {dna_true.shape[1]}")
    
    # Compute comparison metrics
    results = [
        compute_metrics("RNA", "CVAE", rna_true, rna_cvae_pred),
        compute_metrics("RNA", "Mean Imputation", rna_true, rna_mean_pred),
        compute_metrics("RNA", "k-NN Imputation", rna_true, rna_knn_pred),
        compute_metrics("DNA methylation", "CVAE", dna_true, dna_cvae_pred),
        compute_metrics("DNA methylation", "Mean Imputation", dna_true, dna_mean_pred),
        compute_metrics("DNA methylation", "k-NN Imputation", dna_true, dna_knn_pred)
    ]
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("COMPARISON RESULTS: CVAE vs MEAN vs k-NN IMPUTATION")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Create visualizations
    create_comparison_plot(rna_true, rna_cvae_pred, rna_mean_pred, rna_knn_pred,
                          dna_true, dna_cvae_pred, dna_mean_pred, dna_knn_pred, output_dir)
    
    # Create interactive plot
    interactive_fig = create_interactive_plot(rna_true, rna_cvae_pred, rna_mean_pred, rna_knn_pred,
                                            dna_true, dna_cvae_pred, dna_mean_pred, dna_knn_pred)
    
    interactive_fig.write_html(os.path.join(output_dir, 'cvae_interactive_comparison.html'))
    print(f"Interactive plot saved to {os.path.join(output_dir, 'cvae_interactive_comparison.html')}")
    
    # Create metrics bar chart
    create_metrics_bar_chart(results_df, output_dir)
    
    # Save results
    save_results(results_df, output_dir)
    
    print(f"\n✓ CVAE comparison complete! All results saved to {output_dir}/")


if __name__ == "__main__":
    main()

