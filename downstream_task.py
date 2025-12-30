"""
Downstream task: Primary site classification using RNA and estimated DNA methylation
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import Config
from evaluate import get_run_id, load_model_and_data
from src.data import MultiModalDataset


def generate_estimated_dna(vae_model, rna_data, dna_data, labels):
    """Generates estimated DNA methylation data from RNA data"""
    print("Generating estimated DNA methylation data...")
    dataset = MultiModalDataset.from_numpy(rna_data, dna_data, labels)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    est_dna_batches = []
    with torch.no_grad():
        for tpm, _, _ in dataloader:
            tpm = tpm.to(Config.DEVICE)
            _, est_dna, _, _, _ = vae_model(a=tpm)
            est_dna_batches.append(est_dna.cpu().numpy())
            
    return np.concatenate(est_dna_batches, axis=0)


def generate_estimated_rna(vae_model, rna_data, dna_data, labels):
    """Generates estimated RNA data from DNA data."""
    print("Generating estimated RNA data...")
    dataset = MultiModalDataset.from_numpy(rna_data, dna_data, labels)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    est_rna_batches = []
    with torch.no_grad():
        for _, beta_data, _ in dataloader:
            beta_data = beta_data.to(Config.DEVICE)
            est_rna, _, _, _, _ = vae_model(b=beta_data)
            est_rna_batches.append(est_rna.cpu().numpy())

    return np.concatenate(est_rna_batches, axis=0)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Use LayerNorm instead of BatchNorm to avoid issues with small batches
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
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

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_normalized, labels, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=labels
    )

    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = SimpleMLP(features.shape[1], n_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(Config.DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    num_epochs = 100  # Increased from 20

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(Config.DEVICE), batch_labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(Config.DEVICE)
                batch_labels = batch_labels.to(Config.DEVICE)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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


def plot_comparison(metrics_dict, run_id):
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
    width = 0.1  # Narrower bars to fit more scenarios
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (scenario, scores) in enumerate(scenarios.items()):
        ax.bar(x + (i - len(scenarios) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('Scores')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    plot_path = f'plots/downstream_task/downstream_comparison{run_suffix}.png'
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")


def plot_per_tissue_comparison(metrics_dict, le_new, run_id):
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
    width = 0.1  # Narrower bars to fit more scenarios
    fig, ax = plt.subplots(figsize=(18, 10))

    for i, (scenario, scores) in enumerate(f1_scores.items()):
        ax.bar(x + (i - len(f1_scores) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('F1-score')
    ax.set_title('Per-Tissue F1-Score Comparison (Tissues with F1 > 0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    plot_path = f'plots/downstream_task/per_tissue_f1_comparison{run_suffix}.png'
    plt.savefig(plot_path)
    print(f"Per-tissue F1 comparison plot saved to {plot_path}")


if __name__ == "__main__":
    run_id = get_run_id()
    vae_model, val_dataloader, run_id_from_load = load_model_and_data()
    
    # Use run_id from load if available, otherwise use the one from get_run_id()
    if run_id_from_load:
        run_id = run_id_from_load

    # Load validation data separately to get the dataframe
    merged_df = pd.read_pickle('data/processed_data.pkl')
    _, val_df = train_test_split(
        merged_df, 
        test_size=Config.TRAIN_TEST_SPLIT, 
        random_state=Config.RANDOM_SEED
    )
    
    # Create dataset to access dataframe
    val_dataset = MultiModalDataset(val_df)

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

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    est_dna_data = generate_estimated_dna(vae_model, rna_data, dna_data, labels)
    est_rna_data = generate_estimated_rna(vae_model, rna_data, dna_data, labels)

    scenarios = {
        "Orig. RNA": rna_data,
        "Orig. DNA": dna_data,
        "Orig. RNA + Est. DNA": np.concatenate([rna_data, est_dna_data], axis=1),
        "Orig. DNA + Est. RNA": np.concatenate([dna_data, est_rna_data], axis=1),
        "Orig. RNA + Orig. DNA": np.concatenate([rna_data, dna_data], axis=1),
        "Est. DNA": est_dna_data,
        "Est. RNA": est_rna_data,
        "Est. RNA + Est. DNA": np.concatenate([est_rna_data, est_dna_data], axis=1),
    }

    metrics_dict = {}
    for name, data in scenarios.items():
        metrics_dict[name] = run_classification_scenario(data, labels, n_classes, class_weights, name, le_new)

    plot_comparison(metrics_dict, run_id)
    plot_per_tissue_comparison(metrics_dict, le_new, run_id)

    print("\n" + "=" * 50)
    print("Downstream task complete.")
    print("=" * 50)
