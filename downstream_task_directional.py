"""
Downstream task: Primary site classification using RNA and estimated DNA methylation
Using directional VAE models (RNA2DNAVAE and DNA2RNAVAE)
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from src.config import Config
from src.models import RNA2DNAVAE, DNA2RNAVAE
from src.data import MultiModalDataset


def get_run_ids():
    """Get run IDs for both directional models"""
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
    """Load trained directional models and validation data"""
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
            print(f"✓ RNA2DNAVAE model loaded successfully")
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
            print(f"✓ DNA2RNAVAE model loaded successfully")
        else:
            print(f"Warning: Model file {model_path} not found!")
            dna2rna_model = None
    else:
        print("Warning: No DNA2RNAVAE run ID found. Please train the model first.")
    
    return rna2dna_model, dna2rna_model, val_dataloader, val_dataset, rna2dna_run_id, dna2rna_run_id


def generate_estimated_dna(rna2dna_model, rna_data, dna_data, labels):
    """Generates estimated DNA methylation data from RNA data using RNA2DNAVAE"""
    print("Generating estimated DNA methylation data using RNA2DNAVAE...")
    dataset = MultiModalDataset.from_numpy(rna_data, dna_data, labels)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    est_dna_batches = []
    with torch.no_grad():
        for tpm, _, site in dataloader:
            tpm = tpm.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            # Predict DNA from RNA + site
            recon_dna, _, _ = rna2dna_model(rna=tpm, site=site)
            est_dna_batches.append(recon_dna.cpu().numpy())
            
    return np.concatenate(est_dna_batches, axis=0)


def generate_estimated_rna(dna2rna_model, rna_data, dna_data, labels):
    """Generates estimated RNA data from DNA data using DNA2RNAVAE"""
    print("Generating estimated RNA data using DNA2RNAVAE...")
    dataset = MultiModalDataset.from_numpy(rna_data, dna_data, labels)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    est_rna_batches = []
    with torch.no_grad():
        for _, beta_data, site in dataloader:
            beta_data = beta_data.to(Config.DEVICE)
            site = site.to(Config.DEVICE)
            # Predict RNA from DNA + site
            recon_rna, _, _ = dna2rna_model(dna=beta_data, site=site)
            est_rna_batches.append(recon_rna.cpu().numpy())

    return np.concatenate(est_rna_batches, axis=0)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def run_classification_scenario(features, labels, n_classes, class_weights, scenario_name, le_new):
    """Trains and evaluates a classifier for a given scenario."""
    print("\n" + "=" * 50)
    print(f"Scenario: {scenario_name}")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=labels
    )

    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = SimpleMLP(features.shape[1], n_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(Config.DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(Config.DEVICE), batch_labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/20], Loss: {loss.item():.4f}")

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(Config.DEVICE)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_labels.cpu().numpy())

    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, target_names=le_new.classes_, labels=np.arange(len(le_new.classes_)), output_dict=True, zero_division=0
    )
    print(classification_report(
        y_true, y_pred, target_names=le_new.classes_, labels=np.arange(len(le_new.classes_)), zero_division=0
    ))
    return report


def plot_comparison(metrics_dict, rna2dna_run_id, dna2rna_run_id):
    """Plots a comparison of all scenarios."""
    labels = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-score']
    scenarios = {}
    for name, metrics in metrics_dict.items():
        scenarios[name] = [
            metrics['accuracy'],
            metrics['weighted avg']['precision'],
            metrics['weighted avg']['recall'],
            metrics['weighted avg']['f1-score'],
        ]

    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (scenario, scores) in enumerate(scenarios.items()):
        ax.bar(x + (i - len(scenarios) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('Scores')
    ax.set_title('Classifier Performance Comparison (Directional VAE Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task_directional", exist_ok=True)
    plot_path = f'plots/downstream_task_directional/downstream_comparison_rna2dna_{rna2dna_run_id}_dna2rna_{dna2rna_run_id}.png'
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")


def plot_per_tissue_comparison(metrics_dict, le_new, rna2dna_run_id, dna2rna_run_id):
    """Plots a per-tissue F1-score comparison."""
    all_labels = le_new.classes_
    f1_scores = {name: [metrics[label]['f1-score'] for label in all_labels if label in metrics] for name, metrics in metrics_dict.items()}

    # Filter out tissues with 0 F1-score across all scenarios
    non_zero_indices = [i for i, label in enumerate(all_labels) if any(f1_scores[name][i] > 0 for name in f1_scores if i < len(f1_scores[name]))]
    if not non_zero_indices:
        print("No tissues with F1-score > 0 to plot.")
        return

    labels = [all_labels[i] for i in non_zero_indices]
    for name in f1_scores:
        f1_scores[name] = [f1_scores[name][i] for i in non_zero_indices]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 10))

    for i, (scenario, scores) in enumerate(f1_scores.items()):
        ax.bar(x + (i - len(f1_scores) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('F1-score')
    ax.set_title('Per-Tissue F1-Score Comparison (Directional VAE, Tissues with F1 > 0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task_directional", exist_ok=True)
    plot_path = f'plots/downstream_task_directional/per_tissue_f1_comparison_rna2dna_{rna2dna_run_id}_dna2rna_{dna2rna_run_id}.png'
    plt.savefig(plot_path)
    print(f"Per-tissue F1 comparison plot saved to {plot_path}")


if __name__ == "__main__":
    print("="*80)
    print("DOWNSTREAM TASK: PRIMARY SITE CLASSIFICATION")
    print("Using Directional VAE Models (RNA2DNAVAE + DNA2RNAVAE)")
    print("="*80)
    
    # Load models and data
    rna2dna_model, dna2rna_model, val_dataloader, val_dataset, rna2dna_run_id, dna2rna_run_id = load_models_and_data()
    
    if rna2dna_model is None or dna2rna_model is None:
        print("\nError: Both directional models must be trained first!")
        print("Please run train_rna2dna.py and train_dna2rna.py before running this script.")
        exit(1)

    # Extract data from the validation dataset
    val_df = val_dataset.dataframe

    # Filter out classes with fewer than 2 samples in the validation set
    class_counts = val_df['primary_site_encoded'].value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index
    filtered_df = val_df[val_df['primary_site_encoded'].isin(classes_to_keep)].copy()

    # Re-encode labels to be contiguous
    le_new = LabelEncoder()
    filtered_df['primary_site_encoded'] = le_new.fit_transform(filtered_df['primary_site'])

    rna_data = np.array(filtered_df['tpm_unstranded'].tolist()).astype(np.float32)
    dna_data = np.array(filtered_df['beta_value'].tolist()).astype(np.float32)
    labels = filtered_df['primary_site_encoded'].values
    n_classes = len(le_new.classes_)

    print(f"\nData summary:")
    print(f"  Number of samples: {len(rna_data)}")
    print(f"  Number of classes: {n_classes}")
    print(f"  RNA features: {rna_data.shape[1]}")
    print(f"  DNA features: {dna_data.shape[1]}")

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    # Generate estimated data using directional models
    print("\n" + "="*80)
    print("GENERATING ESTIMATED DATA")
    print("="*80)
    est_dna_data = generate_estimated_dna(rna2dna_model, rna_data, dna_data, labels)
    est_rna_data = generate_estimated_rna(dna2rna_model, rna_data, dna_data, labels)
    print(f"✓ Estimated DNA shape: {est_dna_data.shape}")
    print(f"✓ Estimated RNA shape: {est_rna_data.shape}")

    # Define classification scenarios
    scenarios = {
        "Orig. RNA": rna_data,
        "Orig. RNA + Est. DNA": np.concatenate([rna_data, est_dna_data], axis=1),
        "Orig. DNA + Est. RNA": np.concatenate([dna_data, est_rna_data], axis=1),
        "Orig. RNA + Orig. DNA": np.concatenate([rna_data, dna_data], axis=1),
    }

    print("\n" + "="*80)
    print("CLASSIFICATION SCENARIOS")
    print("="*80)
    metrics_dict = {}
    for name, data in scenarios.items():
        metrics_dict[name] = run_classification_scenario(data, labels, n_classes, class_weights, name, le_new)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_comparison(metrics_dict, rna2dna_run_id, dna2rna_run_id)
    plot_per_tissue_comparison(metrics_dict, le_new, rna2dna_run_id, dna2rna_run_id)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nPerformance Summary:")
    for scenario_name, report in metrics_dict.items():
        print(f"\n{scenario_name}:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        print(f"  Weighted Precision: {report['weighted avg']['precision']:.4f}")
        print(f"  Weighted Recall: {report['weighted avg']['recall']:.4f}")

    print("\n" + "="*80)
    print("Downstream task complete!")
    print(f"Results saved to plots/downstream_task_directional/")
    print("="*80)

