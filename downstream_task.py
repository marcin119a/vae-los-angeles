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
from sklearn.model_selection import train_test_split, StratifiedKFold
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


def train_and_evaluate_fold(X_train, X_val, y_train, y_val, input_dim, n_classes, class_weights, le_new):
    """Trains and evaluates a model for a single fold."""
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = SimpleMLP(input_dim, n_classes).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(Config.DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    num_epochs = 100

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(Config.DEVICE), batch_labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(Config.DEVICE)
                batch_labels = batch_labels.to(Config.DEVICE)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(Config.DEVICE)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_labels.cpu().numpy())

    report = classification_report(
        y_true, y_pred, target_names=le_new.classes_, labels=np.arange(len(le_new.classes_)), output_dict=True, zero_division=0
    )
    return report


def run_classification_scenario(features, labels, n_classes, class_weights, scenario_name, le_new, n_folds=5):
    """Trains and evaluates a classifier for a given scenario using cross-validation."""
    print("\n" + "=" * 50)
    print(f"Scenario: {scenario_name}")
    print("=" * 50)

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    
    fold_reports = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features_normalized, labels)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        X_train, X_val = features_normalized[train_idx], features_normalized[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Compute class weights for this fold
        fold_class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        report = train_and_evaluate_fold(
            X_train, X_val, y_train, y_val, 
            features.shape[1], n_classes, fold_class_weights, le_new
        )
        fold_reports.append(report)
    
    # Aggregate results across folds
    aggregated_report = {}
    
    # Aggregate accuracy
    accuracies = [r['accuracy'] for r in fold_reports]
    aggregated_report['accuracy'] = np.mean(accuracies)
    aggregated_report['accuracy_std'] = np.std(accuracies)
    
    # Aggregate weighted averages
    weighted_precisions = [r['weighted avg']['precision'] for r in fold_reports]
    weighted_recalls = [r['weighted avg']['recall'] for r in fold_reports]
    weighted_f1s = [r['weighted avg']['f1-score'] for r in fold_reports]
    
    aggregated_report['weighted avg'] = {
        'precision': np.mean(weighted_precisions),
        'precision_std': np.std(weighted_precisions),
        'recall': np.mean(weighted_recalls),
        'recall_std': np.std(weighted_recalls),
        'f1-score': np.mean(weighted_f1s),
        'f1-score_std': np.std(weighted_f1s),
    }
    
    # Aggregate per-class metrics
    all_classes = le_new.classes_
    for class_name in all_classes:
        if class_name in fold_reports[0]:
            class_precisions = [r[class_name]['precision'] for r in fold_reports if class_name in r]
            class_recalls = [r[class_name]['recall'] for r in fold_reports if class_name in r]
            class_f1s = [r[class_name]['f1-score'] for r in fold_reports if class_name in r]
            
            if class_precisions:
                aggregated_report[class_name] = {
                    'precision': np.mean(class_precisions),
                    'precision_std': np.std(class_precisions),
                    'recall': np.mean(class_recalls),
                    'recall_std': np.std(class_recalls),
                    'f1-score': np.mean(class_f1s),
                    'f1-score_std': np.std(class_f1s),
                }
    
    # Print summary
    print(f"\nCross-Validation Results ({n_folds} folds):")
    print(f"Accuracy: {aggregated_report['accuracy']:.4f} ± {aggregated_report['accuracy_std']:.4f}")
    print(f"Weighted Precision: {aggregated_report['weighted avg']['precision']:.4f} ± {aggregated_report['weighted avg']['precision_std']:.4f}")
    print(f"Weighted Recall: {aggregated_report['weighted avg']['recall']:.4f} ± {aggregated_report['weighted avg']['recall_std']:.4f}")
    print(f"Weighted F1-score: {aggregated_report['weighted avg']['f1-score']:.4f} ± {aggregated_report['weighted avg']['f1-score_std']:.4f}")
    
    return aggregated_report


def plot_comparison(metrics_dict, run_id):
    """Plots a comparison of all scenarios with error bars."""
    labels = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-score']
    scenarios = {}
    errors = {}
    for name, metrics in metrics_dict.items():
        scenarios[name] = [
            metrics['accuracy'],
            metrics['weighted avg']['precision'],
            metrics['weighted avg']['recall'],
            metrics['weighted avg']['f1-score'],
        ]
        errors[name] = [
            metrics.get('accuracy_std', 0),
            metrics['weighted avg'].get('precision_std', 0),
            metrics['weighted avg'].get('recall_std', 0),
            metrics['weighted avg'].get('f1-score_std', 0),
        ]

    x = np.arange(len(labels))
    width = 0.1  # Narrower bars to fit more scenarios
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (scenario, scores) in enumerate(scenarios.items()):
        x_pos = x + (i - len(scenarios) / 2) * width
        ax.bar(x_pos, scores, width, label=scenario, yerr=errors[scenario], capsize=3, alpha=0.8)

    ax.set_ylabel('Scores')
    ax.set_title('Classifier Performance Comparison (Cross-Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    plot_path = f'plots/downstream_task/downstream_comparison{run_suffix}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")


def plot_per_tissue_comparison(metrics_dict, le_new, run_id):
    """Plots a per-tissue F1-score comparison with error bars."""
    # Mapping of full class names to short labels
    class_short_labels = {
        "Hematopoietic and reticuloendothelial systems": "Hemato",
        "Bronchus and lung": "Lung",
        "Breast": "Breast",
        "Kidney": "Kidney",
        "Brain": "Brain",
        "Colon": "Colon",
        "Corpus uteri": "Corpus",
        "Skin": "Skin",
        "Prostate gland": "Prostate",
        "Stomach": "Stomach",
        "Bladder": "Bladder",
        "Liver and intrahepatic bile ducts": "Liver",
        "Pancreas": "Pancreas",
        "Ovary": "Ovary",
        "Uterus, NOS": "Uterus",
        "Cervix uteri": "Cervix",
        "Esophagus": "Esophagus",
        "Adrenal gland": "Adrenal",
        "Other and ill-defined sites": "Other",
        "Other and unspecified parts of tongue": "Tongue",
        "Connective, subcutaneous and other soft tissues": "Connective",
        "Larynx": "Larynx",
        "Rectum": "Rectum",
        "Other and ill-defined sites in lip, oral cavity and pharynx": "Oral/Pharynx"
    }
    
    all_labels = le_new.classes_
    f1_scores = {}
    f1_errors = {}
    
    for name, metrics in metrics_dict.items():
        f1_scores[name] = []
        f1_errors[name] = []
        for label in all_labels:
            if label in metrics:
                f1_scores[name].append(metrics[label]['f1-score'])
                f1_errors[name].append(metrics[label].get('f1-score_std', 0))
            else:
                f1_scores[name].append(0)
                f1_errors[name].append(0)

    # Filter out tissues with 0 F1-score across all scenarios
    non_zero_indices = [i for i, label in enumerate(all_labels) if any(f1_scores[name][i] > 0 for name in f1_scores if i < len(f1_scores[name]))]
    if not non_zero_indices:
        print("No tissues with F1-score > 0 to plot.")
        return

    labels = [all_labels[i] for i in non_zero_indices]
    # Shorten labels using mapping
    short_labels = [class_short_labels.get(label, label) for label in labels]
    
    for name in f1_scores:
        f1_scores[name] = [f1_scores[name][i] for i in non_zero_indices]
        f1_errors[name] = [f1_errors[name][i] for i in non_zero_indices]

    # Split labels into two halves for two panels
    n_labels = len(labels)
    mid_point = n_labels // 2
    
    labels_top = labels[:mid_point]
    short_labels_top = short_labels[:mid_point]
    labels_bottom = labels[mid_point:]
    short_labels_bottom = short_labels[mid_point:]
    
    # Prepare data for top and bottom panels
    f1_scores_top = {}
    f1_errors_top = {}
    f1_scores_bottom = {}
    f1_errors_bottom = {}
    
    for name in f1_scores:
        f1_scores_top[name] = f1_scores[name][:mid_point]
        f1_errors_top[name] = f1_errors[name][:mid_point]
        f1_scores_bottom[name] = f1_scores[name][mid_point:]
        f1_errors_bottom[name] = f1_errors[name][mid_point:]
    
    width = 0.15  # Increased bar width for better readability
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(24, 16))  # Two panels, larger figure size
    
    # Plot top panel
    x_top = np.arange(len(labels_top))
    for i, (scenario, scores) in enumerate(f1_scores_top.items()):
        x_pos = x_top + (i - len(f1_scores_top) / 2) * width
        ax_top.bar(x_pos, scores, width, label=scenario, yerr=f1_errors_top[scenario], capsize=3, alpha=0.8)
    
    ax_top.set_ylabel('F1-score', fontsize=12)
    ax_top.set_title('Per-Tissue F1-Score Comparison (Cross-Validation, Tissues with F1 > 0) - Part 1', fontsize=14)
    ax_top.set_xticks(x_top)
    ax_top.set_xticklabels(short_labels_top, rotation=90, fontsize=10)
    ax_top.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_top.grid(axis='y', alpha=0.3)
    
    # Plot bottom panel
    x_bottom = np.arange(len(labels_bottom))
    for i, (scenario, scores) in enumerate(f1_scores_bottom.items()):
        x_pos = x_bottom + (i - len(f1_scores_bottom) / 2) * width
        ax_bottom.bar(x_pos, scores, width, label=scenario, yerr=f1_errors_bottom[scenario], capsize=3, alpha=0.8)
    
    ax_bottom.set_ylabel('F1-score', fontsize=12)
    ax_bottom.set_title('Per-Tissue F1-Score Comparison (Cross-Validation, Tissues with F1 > 0) - Part 2', fontsize=14)
    ax_bottom.set_xticks(x_bottom)
    ax_bottom.set_xticklabels(short_labels_bottom, rotation=90, fontsize=10)
    ax_bottom.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_bottom.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    os.makedirs("plots/downstream_task", exist_ok=True)
    run_suffix = f"_{run_id}" if run_id else ""
    plot_path = f'plots/downstream_task/per_tissue_f1_comparison{run_suffix}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
