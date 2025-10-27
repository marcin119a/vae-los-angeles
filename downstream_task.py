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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from src.config import Config
from src.models import MultiModalVAE
from compare_vae_vs_mean import load_model_and_data


def get_data():
    """Loads and preprocesses data."""
    print("Loading data...")
    merged_df = pd.read_pickle('data/processed_data.pkl')
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    class_counts = merged_df['primary_site_encoded'].value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index
    filtered_df = merged_df[merged_df['primary_site_encoded'].isin(classes_to_keep)]

    le_new = LabelEncoder()
    filtered_df['primary_site_encoded'] = le_new.fit_transform(filtered_df['primary_site'])

    print(f"Original number of classes: {len(label_encoder.classes_)}")
    print(f"Number of classes after filtering: {len(le_new.classes_)}")

    return filtered_df, le_new, label_encoder

def generate_estimated_dna(vae_model, rna_data):
    """Generates estimated DNA methylation data from RNA data"""
    print("Generating estimated DNA methylation data...")
    with torch.no_grad():
        rna_tensor = torch.tensor(rna_data).to(Config.DEVICE)
        _, est_dna, _, _, _ = vae_model(a=rna_tensor)
    return est_dna.cpu().numpy()

def generate_estimated_rna(vae_model, dna_data):
    """Generates estimated RNA data from DNA data."""
    print("Generating estimated RNA data...")
    with torch.no_grad():
        dna_tensor = torch.tensor(dna_data).to(Config.DEVICE)
        est_rna, _, _, _, _ = vae_model(b=dna_tensor)
    return est_rna.cpu().numpy()


class SimpleMLP(nn.Module):
    """A simple MLP for classification."""
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
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (scenario, scores) in enumerate(scenarios.items()):
        ax.bar(x + (i - len(scenarios) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('Scores')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    plot_path = f'plots/downstream_task/downstream_comparison_{run_id}.png'
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
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 10))

    for i, (scenario, scores) in enumerate(f1_scores.items()):
        ax.bar(x + (i - len(f1_scores) / 2) * width, scores, width, label=scenario)

    ax.set_ylabel('F1-score')
    ax.set_title('Per-Tissue F1-Score Comparison (Tissues with F1 > 0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    plot_path = f'plots/downstream_task/per_tissue_f1_comparison_{run_id}.png'
    plt.savefig(plot_path)
    print(f"Per-tissue F1 comparison plot saved to {plot_path}")


if __name__ == "__main__":
    vae_model, _, _, run_id = load_model_and_data()

    filtered_df, le_new, original_le = get_data()

    rna_data = np.array(filtered_df['tpm_unstranded'].tolist()).astype(np.float32)
    dna_data = np.array(filtered_df['beta_value'].tolist()).astype(np.float32)
    labels = filtered_df['primary_site_encoded'].values
    n_classes = len(le_new.classes_)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    est_dna_data = generate_estimated_dna(vae_model, rna_data)
    est_rna_data = generate_estimated_rna(vae_model, dna_data)

    scenarios = {
        "Orig. RNA": rna_data,
        "Orig. RNA + Est. DNA": np.concatenate([rna_data, est_dna_data], axis=1),
        "Orig. DNA + Est. RNA": np.concatenate([dna_data, est_rna_data], axis=1),
        "Orig. RNA + Orig. DNA": np.concatenate([rna_data, dna_data], axis=1),
    }

    metrics_dict = {}
    for name, data in scenarios.items():
        metrics_dict[name] = run_classification_scenario(data, labels, n_classes, class_weights, name, le_new)

    plot_comparison(metrics_dict, run_id)
    plot_per_tissue_comparison(metrics_dict, le_new, run_id)

    print("\n" + "=" * 50)
    print("Downstream task complete.")
    print("=" * 50)
