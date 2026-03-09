import sys
import os
import pandas as pd
import numpy as np

train_df = pd.read_pickle('data/processed_data.pkl')

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

train_subset, test_subset = train_test_split(train_df, test_size=0.1, random_state=42)
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)

acc_results = []
def evaluate_reconstruction(true_vals, pred_vals, method, target_modality):
    t_flat = true_vals.flatten()
    p_flat = pred_vals.flatten()
    mse = np.mean((t_flat - p_flat)**2)
    pearson, _ = pearsonr(t_flat, p_flat)
    acc_results.append({
        'Method': method,
        'Target': target_modality,
        'MSE': mse,
        'Pearson r': pearson
    })

eval_vae_script = """
import sys
import os
import pandas as pd
import numpy as np
import torch
import json
import glob
from scipy.stats import pearsonr

sys.path.append(os.path.abspath('.'))
from src.models import RNA2DNAVAE
from src.config import Config
import pickle

test_subset = pd.read_pickle('data/temp_test_subset.pkl')
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
Config.DEVICE = device

with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

if 'primary_site_encoded' in test_subset.columns:
    site_labels = test_subset['primary_site_encoded'].values
else:
    site_labels = label_encoder.transform(test_subset['primary_site'])

model = RNA2DNAVAE(
    Config.INPUT_DIM_A, 
    Config.INPUT_DIM_B, 
    len(label_encoder.classes_), 
    Config.LATENT_DIM
).to(device)

def get_run_id():
    if os.path.exists('latest_rna2dna_run_id.txt'):
        with open('latest_rna2dna_run_id.txt', 'r') as f:
            return f.read().strip()
    return None

run_id = get_run_id()
model_path = os.path.join(Config.CHECKPOINT_DIR, f"best_rna2dna_{run_id}.pt") if run_id else None

if model_path and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rna_t = torch.tensor(true_rna_te, dtype=torch.float32).to(device)
    site_t = torch.tensor(site_labels).long().to(device)
    
    with torch.no_grad():
        pred_dna, _, _ = model(rna=rna_t, site=site_t)
        pred_dna = pred_dna.cpu().numpy()

    t_flat = true_dna_te.flatten()
    p_flat = pred_dna.flatten()
    mse = float(np.mean((t_flat - p_flat)**2))
    pearson = float(pearsonr(t_flat, p_flat)[0])

    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'MSE': mse, 'Pearson r': pearson}, f)
else:
    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'error': f'No best model {model_path} found'}, f)
"""
import subprocess
import json

test_subset.to_pickle('data/temp_test_subset.pkl')

with open('temp_vae_eval.py', 'w') as f:
    f.write(eval_vae_script)

try:
    subprocess.run(['venv/bin/python', 'temp_vae_eval.py'], check=True, capture_output=True)
    if os.path.exists('data/temp_vae_acc.json'):
        with open('data/temp_vae_acc.json', 'r') as f:
            vae_results = json.load(f)
        if 'error' not in vae_results:
            print(f"VAE Processed: MSE={vae_results['MSE']:.4f}")
            acc_results.append({
                'Method': 'VAE',
                'Target': 'DNA',
                'MSE': vae_results['MSE'],
                'Pearson r': vae_results['Pearson r']
            })
        else:
            print(f"Error internally: {vae_results['error']}")
except subprocess.CalledProcessError as e:
    print(f"Exception! stdout: {e.stdout.decode()}")
    print(f"Exception! stderr: {e.stderr.decode()}")
finally:
    if os.path.exists('data/temp_test_subset.pkl'): os.remove('data/temp_test_subset.pkl')
    if os.path.exists('temp_vae_eval.py'): os.remove('temp_vae_eval.py')

print(acc_results)

