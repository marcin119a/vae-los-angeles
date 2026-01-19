"""
Script to cross-validate DNA methylation -> RNA transcription and RNA -> DNA models using k-Nearest Neighbors (kNN).
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
from src.config import Config
from src.models import DNA2RNAVAE, RNA2DNAVAE, DNA2RNAAE, RNA2DNAAE
from src.utils.directional_losses import dna2rna_loss, rna2dna_loss
from src.utils.ae_losses import dna2rna_ae_loss, rna2dna_ae_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validate DNA/RNA models using kNN and VAEs.")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--subset", type=float, default=0.1, help="Fraction of data to use (default: 0.1)")
    parser.add_argument("--neighbors", type=int, nargs='+', default=[5, 10], help="List of k values to test (default: 5 10)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of VAE training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--data_path", type=str, default="data/processed_data.pkl", help="Path to processed data pickle")
    return parser.parse_args()

def load_data(data_path, subset_fraction):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_pickle(data_path)
    
    # Use a subset if requested
    if subset_fraction < 1.0:
        print(f"Subsetting data to {subset_fraction*100}%...")
        df = df.sample(frac=subset_fraction, random_state=42)
    
    print(f"Data shape: {df.shape}")
    
    # Extract arrays
    rna_data = np.array(df['tpm_unstranded'].tolist()).astype(np.float32)
    dna_data = np.array(df['beta_value'].tolist()).astype(np.float32)
    site_data = np.array(df['primary_site_encoded'].tolist()).astype(np.int64)
    
    return rna_data, dna_data, site_data

class MeanRegressor:
    """
    Baseline model that predicts the mean of the training data.
    """
    def __init__(self):
        self.mean_vector = None

    def fit(self, X, y):
        self.mean_vector = np.mean(y, axis=0)

    def predict(self, X):
        return np.tile(self.mean_vector, (X.shape[0], 1))

def calculate_metrics(y_true, y_pred):
    """
    Calculate R2, MSE, MAE, Cosine Similarity, and Pearson Correlation.
    """
    # R2
    mean_r2 = r2_score(y_true, y_pred)
    flat_r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    # MSE & MAE (flattened)
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    
    # Cosine Similarity (averaged over samples)
    # Convert to tensors for easy calculation, or use numpy dot product
    # sklearn cosine_similarity computes pairwise, we want row-wise
    y_true_norm = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    cosine_sim = np.sum(y_true_norm * y_pred_norm, axis=1).mean()
    
    # Pearson Correlation (averaged over samples)
    pearson_scores = []
    for i in range(y_true.shape[0]):
        try:
            r, _ = pearsonr(y_true[i], y_pred[i])
            if not np.isnan(r):
                pearson_scores.append(r)
        except:
            pass
    pearson_mean = np.mean(pearson_scores) if pearson_scores else 0.0

    return {
        "Mean R2": mean_r2,
        "Global R2": flat_r2,
        "MSE": mse,
        "MAE": mae,
        "Cosine Sim": cosine_sim,
        "Pearson": pearson_mean
    }

def train_vae(model_class, input_dim_a, input_dim_b, n_sites, train_data_x, train_data_y, train_site, epochs, batch_size, direction):
    # Split training data into inner train and validation for early stopping/scheduler
    # Use 10% of training data for validation
    x_train, x_val, y_train, y_val, site_train, site_val = train_test_split(
        train_data_x, train_data_y, train_site, test_size=0.1, random_state=42
    )
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train), torch.LongTensor(site_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val), torch.LongTensor(site_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model_class(input_dim_a, input_dim_b, n_sites, Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=Config.LR_SCHEDULER_FACTOR, patience=Config.LR_SCHEDULER_PATIENCE)
    
    # Early stopping parameters
    patience = Config.PATIENCE
    best_val_loss = float('inf')
    trigger_times = 0
    best_model_state = None
    
    # Train
    model.train()
    for epoch in range(epochs):
        beta = min(1.0, epoch / Config.BETA_WARMUP_EPOCHS) * Config.BETA_START
        total_train_loss = 0
        
        # Training loop
        model.train()
        for batch_x, batch_y, batch_site in train_loader:
            batch_x, batch_y, batch_site = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE), batch_site.to(Config.DEVICE)
            
            if direction == "DNA -> RNA":
                # X=DNA, Y=RNA. Model expects (dna, site) -> rna
                recon, mu, logvar = model(dna=batch_x, site=batch_site)
                loss, _, _ = dna2rna_loss(recon, batch_y, mu, logvar, beta=beta)
            else:
                # X=RNA, Y=DNA. Model expects (rna, site) -> dna
                recon, mu, logvar = model(rna=batch_x, site=batch_site)
                loss, _, _ = rna2dna_loss(recon, batch_y, mu, logvar, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_site in val_loader:
                batch_x, batch_y, batch_site = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE), batch_site.to(Config.DEVICE)
                
                if direction == "DNA -> RNA":
                    recon, mu, logvar = model(dna=batch_x, site=batch_site)
                    loss, _, _ = dna2rna_loss(recon, batch_y, mu, logvar, beta=beta)
                else:
                    recon, mu, logvar = model(rna=batch_x, site=batch_site)
                    loss, _, _ = rna2dna_loss(recon, batch_y, mu, logvar, beta=beta)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                # Early stopping
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return model

def train_ae(model_class, input_dim_a, input_dim_b, n_sites, train_data_x, train_data_y, train_site, epochs, batch_size, direction):
    # Split training data into inner train and validation for early stopping/scheduler
    # Use 10% of training data for validation
    x_train, x_val, y_train, y_val, site_train, site_val = train_test_split(
        train_data_x, train_data_y, train_site, test_size=0.1, random_state=42
    )
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train), torch.LongTensor(site_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val), torch.LongTensor(site_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = model_class(input_dim_a, input_dim_b, n_sites, Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=Config.LR_SCHEDULER_FACTOR, patience=Config.LR_SCHEDULER_PATIENCE)
    
    # Early stopping parameters
    patience = Config.PATIENCE
    best_val_loss = float('inf')
    trigger_times = 0
    best_model_state = None
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        
        # Training loop
        model.train()
        for batch_x, batch_y, batch_site in train_loader:
            batch_x, batch_y, batch_site = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE), batch_site.to(Config.DEVICE)
            
            if direction == "DNA -> RNA":
                # X=DNA, Y=RNA. Model expects (dna, site) -> rna
                recon, _ = model(dna=batch_x, site=batch_site)
                loss, _ = dna2rna_ae_loss(recon, batch_y)
            else:
                # X=RNA, Y=DNA. Model expects (rna, site) -> dna
                recon, _ = model(rna=batch_x, site=batch_site)
                loss, _ = rna2dna_ae_loss(recon, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_site in val_loader:
                batch_x, batch_y, batch_site = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE), batch_site.to(Config.DEVICE)
                
                if direction == "DNA -> RNA":
                    recon, _ = model(dna=batch_x, site=batch_site)
                    loss, _ = dna2rna_ae_loss(recon, batch_y)
                else:
                    recon, _ = model(rna=batch_x, site=batch_site)
                    loss, _ = rna2dna_ae_loss(recon, batch_y)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                # Early stopping
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return model

def run_cross_validation(X, y, site, k_values, fold_indices, direction_name, model_type="knn", epochs=10, batch_size=64):
    print(f"\nRunning Cross-Validation for {direction_name} ({model_type})...")
    
    results = []
    
    # Determine params
    if model_type == "knn":
        params_to_test = k_values
        param_name = "k"
    elif model_type == "vae":
        params_to_test = [epochs]
        param_name = "epochs"
    elif model_type == "ae":
        params_to_test = [epochs]
        param_name = "epochs"
    elif model_type == "mean":
        params_to_test = [0] # Dummy param
        param_name = "dummy"
    
    for param in params_to_test:
        if model_type != "mean":
            print(f"  Testing {param_name}={param}...")
        else:
            print(f"  Testing Mean Baseline...")

        # Initialize metrics lists
        fold_metrics = {k: [] for k in ["Mean R2", "Global R2", "MSE", "MAE", "Cosine Sim", "Pearson"]}
        start_time = time.time()
        
        for fold_idx, (train_index, val_index) in enumerate(fold_indices):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            site_train, site_val = site[train_index], site[val_index]
            
            if model_type == "knn":
                model = KNeighborsRegressor(n_neighbors=param, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            elif model_type == "mean":
                model = MeanRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

            elif model_type == "vae":
                # Determine dimensions
                n_sites = int(site.max() + 1)
                
                if direction_name == "DNA -> RNA":
                    input_dim_a = y.shape[1] # RNA
                    input_dim_b = X.shape[1] # DNA
                    model_class = DNA2RNAVAE
                else:
                    input_dim_a = X.shape[1] # RNA
                    input_dim_b = y.shape[1] # DNA
                    model_class = RNA2DNAVAE
                
                model = train_vae(model_class, input_dim_a, input_dim_b, n_sites, 
                                  X_train, y_train, site_train, 
                                  param, batch_size, direction_name)
                
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val).to(Config.DEVICE)
                    site_val_tensor = torch.tensor(site_val).to(Config.DEVICE)
                    
                    if direction_name == "DNA -> RNA":
                        y_pred_tensor, _, _ = model(dna=X_val_tensor, site=site_val_tensor)
                    else:
                        y_pred_tensor, _, _ = model(rna=X_val_tensor, site=site_val_tensor)
                    
                    y_pred = y_pred_tensor.cpu().numpy()

            elif model_type == "ae":
                # Determine dimensions
                n_sites = int(site.max() + 1)
                
                if direction_name == "DNA -> RNA":
                    input_dim_a = y.shape[1] # RNA
                    input_dim_b = X.shape[1] # DNA
                    model_class = DNA2RNAAE
                else:
                    input_dim_a = X.shape[1] # RNA
                    input_dim_b = y.shape[1] # DNA
                    model_class = RNA2DNAAE
                
                model = train_ae(model_class, input_dim_a, input_dim_b, n_sites, 
                                 X_train, y_train, site_train, 
                                 param, batch_size, direction_name)
                
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val).to(Config.DEVICE)
                    site_val_tensor = torch.tensor(site_val).to(Config.DEVICE)
                    
                    if direction_name == "DNA -> RNA":
                        y_pred_tensor, _ = model(dna=X_val_tensor, site=site_val_tensor)
                    else:
                        y_pred_tensor, _ = model(rna=X_val_tensor, site=site_val_tensor)
                    
                    y_pred = y_pred_tensor.cpu().numpy()

            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred)
            for k, v in metrics.items():
                fold_metrics[k].append(v)
            
        elapsed = time.time() - start_time
        
        # Aggregate results
        aggregated_results = {
            "direction": direction_name,
            "model": model_type,
            "param_name": param_name,
            "param_value": param,
            "time": elapsed,
            "fold_metrics": fold_metrics
        }
        
        # Add mean/std for all metrics
        for metric_name in fold_metrics:
            aggregated_results[f"mean_{metric_name}"] = np.mean(fold_metrics[metric_name])
            aggregated_results[f"std_{metric_name}"] = np.std(fold_metrics[metric_name])
            
        print(f"    Mean R2 = {aggregated_results['mean_Mean R2']:.4f} (+/- {aggregated_results['std_Mean R2']:.4f})")
        print(f"    MSE     = {aggregated_results['mean_MSE']:.4f} (+/- {aggregated_results['std_MSE']:.4f})")
        
        results.append(aggregated_results)
        
    return results

def create_plotly_plots(results, output_dir="plots/plotly"):
    print(f"Creating Plotly plots in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of metrics from the first result
    metrics = list(results[0]["fold_metrics"].keys())
    
    for metric in metrics:
        plot_data = []
        for res in results:
            label = f"{res['model']}"
            if res['model'] == 'knn':
                 label += f" (k={res['param_value']})"
            elif res['model'] == 'vae':
                 label += f" (ep={res['param_value']})"
            elif res['model'] == 'ae':
                 label += f" (ep={res['param_value']})"
                 
            for score in res['fold_metrics'][metric]:
                plot_data.append({
                    "Direction": res['direction'],
                    "Model": label,
                    "Score": score
                })
                
        df = pd.DataFrame(plot_data)
        
        fig = px.box(df, x="Model", y="Score", color="Direction", 
                     title=f"Cross-Validation {metric}",
                     points="all", hover_data=["Model", "Direction"])
        fig.update_layout(template="plotly_white")
        
        safe_metric_name = metric.lower().replace(" ", "_")
        fig.write_html(f"{output_dir}/cv_results_{safe_metric_name}.html")
    
    print("Plotly plots saved.")

def perform_statistical_comparison(results, metric="Mean R2"):
    print("\n" + "="*80)
    print(f"STATISTICAL COMPARISON (Paired t-test) on {metric}")
    print("="*80)
    
    # Group results by direction
    directions = set(r['direction'] for r in results)
    
    for direction in directions:
        print(f"\nDirection: {direction}")
        dir_results = [r for r in results if r['direction'] == direction]
        
        # Find best model for each type (knn vs vae) based on mean R2 (always use R2 to pick 'best', but compare on requested metric)
        # Or better: Pick best based on the metric itself? 
        # For error metrics (MSE, MAE), lower is better. For R2/Corr, higher is better.
        # Let's stick to using Mean R2 to select the best configuration, then compare them on the specific metric.
        
        knn_results = [r for r in dir_results if r['model'] == 'knn']
        vae_results = [r for r in dir_results if r['model'] == 'vae']
        ae_results = [r for r in dir_results if r['model'] == 'ae']
        mean_results = [r for r in dir_results if r['model'] == 'mean']
        
        if not knn_results or not vae_results:
            continue
            
        best_knn = max(knn_results, key=lambda x: x['mean_Mean R2'])
        best_vae = max(vae_results, key=lambda x: x['mean_Mean R2'])
        
        # Get scores for the selected metric
        knn_scores = best_knn['fold_metrics'][metric]
        vae_scores = best_vae['fold_metrics'][metric]
        
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(knn_scores, vae_scores)
        
        print(f"  Best kNN: k={best_knn['param_value']} ({metric}={np.mean(knn_scores):.4f})")
        print(f"  Best VAE: epochs={best_vae['param_value']} ({metric}={np.mean(vae_scores):.4f})")
        
        if ae_results:
            best_ae = max(ae_results, key=lambda x: x['mean_Mean R2'])
            ae_scores = best_ae['fold_metrics'][metric]
            print(f"  Best AE: epochs={best_ae['param_value']} ({metric}={np.mean(ae_scores):.4f})")
            
            # Compare AE vs VAE
            t_ae_vae, p_ae_vae = stats.ttest_rel(ae_scores, vae_scores)
            print(f"  AE vs VAE: t={t_ae_vae:.4f}, p={p_ae_vae:.4e}")
            
            # Compare AE vs kNN
            t_ae_knn, p_ae_knn = stats.ttest_rel(ae_scores, knn_scores)
            print(f"  AE vs kNN: t={t_ae_knn:.4f}, p={p_ae_knn:.4e}")
        
        if mean_results:
            mean_baseline = mean_results[0]
            mean_scores = mean_baseline['fold_metrics'][metric]
            print(f"  Mean Baseline: ({metric}={np.mean(mean_scores):.4f})")
            
            # Compare VAE vs Mean
            t_mean, p_mean = stats.ttest_rel(vae_scores, mean_scores)
            print(f"  VAE vs Mean: t={t_mean:.4f}, p={p_mean:.4e}")

        print(f"  VAE vs kNN: t={t_stat:.4f}, p={p_val:.4e}")
        if p_val < 0.05:
             # Determine winner direction
            mean_knn = np.mean(knn_scores)
            mean_vae = np.mean(vae_scores)
            
            # For arrays where higher is better
            higher_better = ["R2", "Cosine", "Pearson"]
            is_higher_better = any(x in metric for x in higher_better)
            
            if is_higher_better:
                winner = "kNN" if mean_knn > mean_vae else "VAE"
            else:
                winner = "kNN" if mean_knn < mean_vae else "VAE"
                
            print(f"  -> Significant difference! {winner} performs better.")
        else:
            print(f"  -> No significant difference detected (p >= 0.05).")

def main():
    args = parse_args()
    
    # Override default epochs if not specified by user (argparse default is 100, we want 200 or user arg)
    # Actually argparse default is 100 in definition. Let's force it to 200 if it matches default, 
    # but the user might have passed 100 explicitly. 
    # Safer to just change the argparse default. But since I can't change argparse definition easily without editing that part...
    # I'll just trust the passed args. 
    # Wait, plan said "Set epochs default to 200". I should update argparse default.
    
    try:
        rna_data, dna_data, site_data = load_data(args.data_path, args.subset)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Generate consistent fold indices
    print(f"\nGenerating {args.folds} folds to be used across all models...")
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_indices = list(kf.split(rna_data))

    all_results = []
    
    # DNA -> RNA
    print("\n--- Processing DNA -> RNA ---")
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, [], fold_indices, "DNA -> RNA", "mean"))
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, args.neighbors, fold_indices, "DNA -> RNA", "knn"))
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, [args.epochs], fold_indices, "DNA -> RNA", "vae", epochs=args.epochs, batch_size=args.batch_size))
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, [args.epochs], fold_indices, "DNA -> RNA", "ae", epochs=args.epochs, batch_size=args.batch_size))
    
    # RNA -> DNA
    print("\n--- Processing RNA -> DNA ---")
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, [], fold_indices, "RNA -> DNA", "mean"))
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, args.neighbors, fold_indices, "RNA -> DNA", "knn"))
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, [args.epochs], fold_indices, "RNA -> DNA", "vae", epochs=args.epochs, batch_size=args.batch_size))
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, [args.epochs], fold_indices, "RNA -> DNA", "ae", epochs=args.epochs, batch_size=args.batch_size))
    
    # Summary
    print("\n" + "="*120)
    print("FINAL RESULTS SUMMARY (Mean R2 & MSE)")
    print("="*120)
    print(f"{'Direction':<12} | {'Model':<5} | {'Param':<8} | {'Mean R2':<10} | {'Std':<8} | {'MSE':<10} | {'Std':<8} | {'Time (s)':<8}")
    print("-" * 120)
    for res in all_results:
        print(f"{res['direction']:<12} | {res['model']:<5} | {res['param_name']}={res['param_value']:<4} | {res['mean_Mean R2']:<10.4f} | {res['std_Mean R2']:<8.4f} | {res['mean_MSE']:<10.4f} | {res['std_MSE']:<8.4f} | {res['time']:<8.2f}")
    print("="*120)
    
    # Statistical Comparison
    # Compare on key metrics
    perform_statistical_comparison(all_results, metric="Mean R2")
    perform_statistical_comparison(all_results, metric="MSE")
    perform_statistical_comparison(all_results, metric="Pearson")
    
    # Create visualizations
    create_plotly_plots(all_results)

if __name__ == "__main__":
    main()
