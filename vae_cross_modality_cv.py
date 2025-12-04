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
from scipy import stats
from src.config import Config
from src.models import DNA2RNAVAE, RNA2DNAVAE
from src.utils.directional_losses import dna2rna_loss, rna2dna_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validate DNA/RNA models using kNN and VAEs.")
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--subset", type=float, default=0.1, help="Fraction of data to use (default: 0.1)")
    parser.add_argument("--neighbors", type=int, nargs='+', default=[5, 10], help="List of k values to test (default: 5 10)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of VAE training epochs (default: 100)")
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

def run_cross_validation(X, y, site, k_values, n_folds, direction_name, model_type="knn", epochs=10, batch_size=64):
    print(f"\nRunning Cross-Validation for {direction_name} ({model_type})...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    # For VAE, we don't iterate over k_values, but we can treat 'epochs' or similar as hyperparam if needed.
    # For now, just run once per fold for VAE.
    params_to_test = k_values if model_type == "knn" else [epochs]
    param_name = "k" if model_type == "knn" else "epochs"
    
    for param in params_to_test:
        print(f"  Testing {param_name}={param}...")
        fold_scores = []
        start_time = time.time()
        
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            site_train, site_val = site[train_index], site[val_index]
            
            if model_type == "knn":
                model = KNeighborsRegressor(n_neighbors=param, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            elif model_type == "vae":
                # Determine dimensions
                # DNA->RNA: X=DNA (input_dim_b), Y=RNA (input_dim_a)
                # RNA->DNA: X=RNA (input_dim_a), Y=DNA (input_dim_b)
                n_sites = int(site.max() + 1)
                
                if direction_name == "DNA -> RNA":
                    input_dim_a = y.shape[1] # RNA
                    input_dim_b = X.shape[1] # DNA
                    model_class = DNA2RNAVAE
                else:
                    input_dim_a = X.shape[1] # RNA
                    input_dim_b = y.shape[1] # DNA
                    model_class = RNA2DNAVAE
                # Train VAE
                # Note: train_vae now handles internal validation split
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

            score = r2_score(y_val, y_pred)
            fold_scores.append(score)
            
        elapsed = time.time() - start_time
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"    {param_name}={param}: Mean R2 = {mean_score:.4f} (+/- {std_score:.4f}) [Time: {elapsed:.2f}s]")
        
        results.append({
            "direction": direction_name,
            "model": model_type,
            "param_name": param_name,
            "param_value": param,
            "mean_r2": mean_score,
            "std_r2": std_score,
            "time": elapsed,
            "fold_scores": fold_scores
        })
        
    return results

def create_cv_boxplot(results, output_file="plots/cv_results_boxplot.png"):
    print(f"Creating boxplot: {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Flatten results for plotting
    plot_data = []
    for res in results:
        label = f"{res['model']} ({res['param_name']}={res['param_value']})"
        for score in res['fold_scores']:
            plot_data.append({
                "Direction": res['direction'],
                "Model": label,
                "R2 Score": score
            })
            
    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="Model", y="R2 Score", hue="Direction", showfliers=False)
    plt.title("Cross-Validation R2 Scores Distribution")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print("Boxplot saved.")

def perform_statistical_comparison(results):
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON (Paired t-test)")
    print("="*80)
    
    # Group results by direction
    directions = set(r['direction'] for r in results)
    
    for direction in directions:
        print(f"\nDirection: {direction}")
        dir_results = [r for r in results if r['direction'] == direction]
        
        # Find best model for each type (knn vs vae) based on mean R2
        knn_results = [r for r in dir_results if r['model'] == 'knn']
        vae_results = [r for r in dir_results if r['model'] == 'vae']
        
        if not knn_results or not vae_results:
            continue
            
        best_knn = max(knn_results, key=lambda x: x['mean_r2'])
        best_vae = max(vae_results, key=lambda x: x['mean_r2'])
        
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(best_knn['fold_scores'], best_vae['fold_scores'])
        
        print(f"  Best kNN: k={best_knn['param_value']} (R2={best_knn['mean_r2']:.4f})")
        print(f"  Best VAE: epochs={best_vae['param_value']} (R2={best_vae['mean_r2']:.4f})")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4e}")
        if p_val < 0.05:
            winner = "kNN" if best_knn['mean_r2'] > best_vae['mean_r2'] else "VAE"
            print(f"  -> Significant difference! {winner} performs better.")
        else:
            print(f"  -> No significant difference detected (p >= 0.05).")

def main():
    args = parse_args()
    
    try:
        rna_data, dna_data, site_data = load_data(args.data_path, args.subset)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    all_results = []
    
    # DNA -> RNA
    # X = DNA, y = RNA
    # kNN
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, args.neighbors, args.folds, "DNA -> RNA", "knn"))
    # VAE
    all_results.extend(run_cross_validation(dna_data, rna_data, site_data, [args.epochs], args.folds, "DNA -> RNA", "vae", epochs=args.epochs, batch_size=args.batch_size))
    
    # RNA -> DNA
    # X = RNA, y = DNA
    # kNN
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, args.neighbors, args.folds, "RNA -> DNA", "knn"))
    # VAE
    all_results.extend(run_cross_validation(rna_data, dna_data, site_data, [args.epochs], args.folds, "RNA -> DNA", "vae", epochs=args.epochs, batch_size=args.batch_size))
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Direction':<15} | {'Model':<5} | {'Param':<10} | {'Mean R2':<10} | {'Std Dev':<10} | {'Time (s)':<10}")
    print("-" * 80)
    for res in all_results:
        print(f"{res['direction']:<15} | {res['model']:<5} | {res['param_name']}={res['param_value']:<4} | {res['mean_r2']:<10.4f} | {res['std_r2']:<10.4f} | {res['time']:<10.2f}")
    print("="*80)
    
    # Statistical Comparison
    perform_statistical_comparison(all_results)
    
    # Create visualization
    create_cv_boxplot(all_results)

if __name__ == "__main__":
    main()
