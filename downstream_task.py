"""
Downstream task: Primary site classification using RNA and estimated DNA methylation
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from src.config import Config
from src.models import MultiModalVAE

# --- 1. Load Data and Pre-trained VAE ---

def load_data():
    """Loads processed data and label encoder"""
    print("Loading data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return merged_df, label_encoder

def load_vae_model(vae_model_path, n_sites):
    """Loads the pre-trained VAE model"""
    print(f"Loading VAE model from: {vae_model_path}")
    model = MultiModalVAE(
        Config.INPUT_DIM_A,
        Config.INPUT_DIM_B,
        n_sites,
        Config.LATENT_DIM
    )
    model.load_state_dict(torch.load(vae_model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

# --- 2. Data Preparation ---

def generate_estimated_dna(vae_model, rna_data):
    """Generates estimated DNA methylation data from RNA data"""
    print("Generating estimated DNA methylation data...")
    with torch.no_grad():
        rna_tensor = torch.tensor(rna_data).to(Config.DEVICE)
        _, est_dna, _, _, _ = vae_model(a=rna_tensor)
    return est_dna.cpu().numpy()

def prepare_dataloaders(features, labels, class_weights):
    """Creates dataloaders for the classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=labels
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, y_test

# --- 3. Classifier Model ---

class SimpleMLP(nn.Module):
    """A simple MLP for classification"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# --- 4. Training and Evaluation ---

def train_classifier(model, train_loader, class_weights, epochs=20):
    """Trains the classifier"""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(Config.DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def evaluate_classifier(model, test_loader, target_names):
    """Evaluates the classifier"""
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(Config.DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=target_names, labels=np.arange(len(target_names)), output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names, labels=np.arange(len(target_names))))
    return report

def plot_comparison(metrics_rna, metrics_est_dna, metrics_orig_dna, run_id):
    """Plots a comparison of the three scenarios"""
    labels = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-score']
    scenarios = {
        'RNA-only': [metrics_rna['accuracy'], metrics_rna['weighted avg']['precision'], metrics_rna['weighted avg']['recall'], metrics_rna['weighted avg']['f1-score']],
        'RNA + Est. DNA': [metrics_est_dna['accuracy'], metrics_est_dna['weighted avg']['precision'], metrics_est_dna['weighted avg']['recall'], metrics_est_dna['weighted avg']['f1-score']],
        'RNA + Orig. DNA': [metrics_orig_dna['accuracy'], metrics_orig_dna['weighted avg']['precision'], metrics_orig_dna['weighted avg']['recall'], metrics_orig_dna['weighted avg']['f1-score']]
    }

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))
    
    i = 0
    for scenario, scores in scenarios.items():
        ax.bar(x + (i - 1) * width, scores, width, label=scenario)
        i += 1

    ax.set_ylabel('Scores')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plot_path = f'plots/downstream_task/downstream_comparison_{run_id}.png'
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

def plot_per_tissue_comparison(metrics_rna, metrics_est_dna, metrics_orig_dna, le_new, run_id):
    """Plots a per-tissue F1-score comparison."""
    labels = le_new.classes_
    rna_f1 = [metrics_rna[label]['f1-score'] for label in labels]
    est_dna_f1 = [metrics_est_dna[label]['f1-score'] for label in labels]
    orig_dna_f1 = [metrics_orig_dna[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - width, rna_f1, width, label='RNA-only')
    rects2 = ax.bar(x, est_dna_f1, width, label='RNA + Est. DNA')
    rects3 = ax.bar(x + width, orig_dna_f1, width, label='RNA + Orig. DNA')

    ax.set_ylabel('F1-score')
    ax.set_title('Per-Tissue F1-Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()

    fig.tight_layout()
    plot_path = f'plots/downstream_task/per_tissue_f1_comparison_{run_id}.png'
    plt.savefig(plot_path)
    print(f"Per-tissue F1 comparison plot saved to {plot_path}")

# --- 5. Main Execution ---

def main():
    """Main function to run the downstream task"""
    # Get run ID
    with open('latest_run_id.txt', 'r') as f:
        run_id = f.read().strip()
    
    vae_model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_multivae_{run_id}.pt')

    # Load data
    merged_df, label_encoder = load_data()

    # --- Pre-processing: Filter small classes ---
    class_counts = merged_df['primary_site_encoded'].value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index
    
    filtered_df = merged_df[merged_df['primary_site_encoded'].isin(classes_to_keep)]
    
    # Re-encode labels
    from sklearn.preprocessing import LabelEncoder
    le_new = LabelEncoder()
    filtered_df['primary_site_encoded'] = le_new.fit_transform(filtered_df['primary_site'])
    
    print(f"Original number of classes: {len(label_encoder.classes_)}")
    print(f"Number of classes after filtering: {len(le_new.classes_)}")

    # Load VAE with the original number of sites, as it was trained on all of them
    vae_model = load_vae_model(vae_model_path, len(label_encoder.classes_))

    # Prepare data
    rna_data = np.array(filtered_df['tpm_unstranded'].tolist()).astype(np.float32)
    dna_data = np.array(filtered_df['beta_value'].tolist()).astype(np.float32)
    labels = filtered_df['primary_site_encoded'].values
    n_sites = len(le_new.classes_)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    
    # --- Scenario 1: RNA-only classification ---
    print("\n" + "="*50)
    print("Scenario 1: RNA-only Classification")
    print("="*50)
    
    train_loader_rna, test_loader_rna, y_test_rna = prepare_dataloaders(rna_data, labels, class_weights)
    
    model_rna = SimpleMLP(Config.INPUT_DIM_A, n_sites).to(Config.DEVICE)
    train_classifier(model_rna, train_loader_rna, class_weights)
    metrics_rna = evaluate_classifier(model_rna, test_loader_rna, le_new.classes_)
    
    # --- Scenario 2: RNA + Estimated DNA classification ---
    print("\n" + "="*50)
    print("Scenario 2: RNA + Estimated DNA Classification")
    print("="*50)
    
    est_dna_data = generate_estimated_dna(vae_model, rna_data)
    rna_plus_est_dna = np.concatenate([rna_data, est_dna_data], axis=1)
    
    train_loader_est, test_loader_est, y_test_est = prepare_dataloaders(rna_plus_est_dna, labels, class_weights)
    
    model_est = SimpleMLP(Config.INPUT_DIM_A + Config.INPUT_DIM_B, n_sites).to(Config.DEVICE)
    train_classifier(model_est, train_loader_est, class_weights)
    metrics_est = evaluate_classifier(model_est, test_loader_est, le_new.classes_)

    # --- Scenario 3: RNA + Original DNA classification ---
    print("\n" + "="*50)
    print("Scenario 3: RNA + Original DNA Classification")
    print("="*50)

    rna_plus_orig_dna = np.concatenate([rna_data, dna_data], axis=1)

    train_loader_orig, test_loader_orig, y_test_orig = prepare_dataloaders(rna_plus_orig_dna, labels, class_weights)

    model_orig = SimpleMLP(Config.INPUT_DIM_A + Config.INPUT_DIM_B, n_sites).to(Config.DEVICE)
    train_classifier(model_orig, train_loader_orig, class_weights)
    metrics_orig = evaluate_classifier(model_orig, test_loader_orig, le_new.classes_)

    # --- Compare results ---
    plot_comparison(metrics_rna, metrics_est, metrics_orig, run_id)
    plot_per_tissue_comparison(metrics_rna, metrics_est, metrics_orig, le_new, run_id)

    print("\n" + "="*50)
    print("Downstream task complete.")
    print("Paper suggestion for similar tasks:")
    print("  - 'Multimodal data integration using machine learning for cancer classification'")
    print("    (Look for papers on PubMed or Google Scholar with similar keywords)")
    print("="*50)

if __name__ == "__main__":
    main()
