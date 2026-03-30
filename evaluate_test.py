
import sys
import os
import pandas as pd
import numpy as np
import torch
import json
import pickle
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from src.config import Config
from src.models import RNA2DNAVAE, DNA2RNAVAE

def get_run_id(modality):
    file_path = f'latest_{modality}_run_id.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None

def evaluate_model(modality='rna2dna', model_type="modified"):
    print(f"\nEvaluating {modality} model ({model_type})...")
    
    # Load data
    train_df = pd.read_pickle('data/processed_data.pkl')
    _, test_subset = train_test_split(train_df, test_size=0.1, random_state=42)
    
    with open('data/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    Config.DEVICE = device
    
    # Initialize model
    if modality == 'rna2dna':
        model = RNA2DNAVAE(Config.INPUT_DIM_A, Config.INPUT_DIM_B, len(label_encoder.classes_), Config.LATENT_DIM).to(device)
        target_name = 'DNA'
        true_vals = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
        input_vals = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)
    else:
        model = DNA2RNAVAE(Config.INPUT_DIM_A, Config.INPUT_DIM_B, len(label_encoder.classes_), Config.LATENT_DIM).to(device)
        target_name = 'RNA'
        true_vals = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)
        input_vals = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
        
    if 'primary_site_encoded' in test_subset.columns:
        site_labels = test_subset['primary_site_encoded'].values
    else:
        site_labels = label_encoder.transform(test_subset['primary_site'])
        
    run_id = get_run_id(modality)
    if not run_id:
        print(f"No run ID found for {modality}")
        return
        
    model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{modality}_{run_id}.pt")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    input_t = torch.tensor(input_vals, dtype=torch.float32).to(device)
    site_t = torch.tensor(site_labels).long().to(device)
    
    with torch.no_grad():
        if modality == 'rna2dna':
            # New forward returns: out_dna, mu, logvar, mu_list, logvar_list
            outputs = model(rna=input_t, site=site_t)
        else:
            # New forward returns: out_rna, mu, logvar, mu_list, logvar_list
            outputs = model(dna=input_t, site=site_t)
            
        pred_vals = outputs[0].cpu().numpy()
        
    mse = np.mean((true_vals.flatten() - pred_vals.flatten())**2)
    pearson, _ = pearsonr(true_vals.flatten(), pred_vals.flatten())
    
    print(f"Results for {modality}: MSE={mse:.4f}, Pearson={pearson:.4f}")
    
    # Save results
    res_file = f'plots/eval_results_{modality}_{run_id}.json'
    res = {
        'Method': 'VAE',
        'Target': target_name,
        'Modality': modality,
        'MSE': float(mse),
        'Pearson r': float(pearson),
        'run_id': run_id,
        'model_type': model_type
    }
    os.makedirs('plots', exist_ok=True)
    with open(res_file, 'w') as f:
        json.dump(res, f)
    print(f"Saved to {res_file}")

if __name__ == "__main__":
    model_type = os.getenv("MODEL_TYPE", "modified")
    evaluate_model('rna2dna', model_type=model_type)
    evaluate_model('dna2rna', model_type=model_type)
