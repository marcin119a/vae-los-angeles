import subprocess
import os
import json

df = __import__('pandas').read_pickle('data/processed_data.pkl')
_, test_subset = __import__('sklearn.model_selection').model_selection.train_test_split(df, test_size=0.1, random_state=42)
test_subset.to_pickle('data/temp_test_subset.pkl')

eval_script = """
import sys
import os

# Insert root at beginning to prioritize root src over MIMIR src for config, etc.
root_path = os.path.abspath('..')
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
import numpy as np
import torch
import json
from scipy.stats import pearsonr

# Import Config from ROOT src
from src.config import Config

# Now we need MIMIR's src.mae_masked, so temporarily restore standard path resolution 
# by moving root_path to the end so local MIMIR imports take precedence
sys.path.remove(root_path)
sys.path.append(root_path)
from src.mae_masked import MultiModalWithSharedSpace

test_subset = pd.read_pickle('../data/temp_test_subset.pkl')
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
cfg = Config()
rna_dim = len(cfg.selected_rna_features)
dna_dim = len(cfg.selected_dna_features)

shared_model = MultiModalWithSharedSpace(
    modalities={'rna': rna_dim, 'dna': dna_dim},
    shared_dim=128, proj_depth=1
).to(device)

shared_model.load_state_dict(torch.load('mimir_checkpoints/finetuned/shared_model_ep100.pt', map_location=device, weights_only=True))
shared_model.eval()

rna_val_t = torch.tensor(true_rna_te, dtype=torch.float32).to(device)
res = []
bs = 256
with torch.no_grad():
    for i in range(0, rna_val_t.size(0), bs):
        batch = rna_val_t[i:i+bs]
        encoded = shared_model.encoders['rna'](batch)
        shared = shared_model.projections['rna'](encoded)
        from_shared = shared_model.rev_projections['dna'](shared)
        imputed_dna = shared_model.decoders['dna'](from_shared)
        res.append(imputed_dna.cpu().numpy())
mimir_preds = np.concatenate(res, axis=0)

t_flat = true_dna_te.flatten()
p_flat = mimir_preds.flatten()
mse = float(np.mean((t_flat - p_flat)**2))
pearson = float(pearsonr(t_flat, p_flat)[0])

with open('../data/temp_mimir_acc.json', 'w') as f:
    json.dump({'MSE': mse, 'Pearson r': pearson}, f)
"""
with open('MIMIR/temp_eval.py', 'w') as f:
    f.write(eval_script)

print("MIMIR TEST:")
try:
    res = subprocess.run(['../venv/bin/python', 'temp_eval.py'], cwd='MIMIR', capture_output=True, text=True, check=True)
    print("MIMIR SUCCESS")
    with open('data/temp_mimir_acc.json', 'r') as f:
        print(json.load(f))
except subprocess.CalledProcessError as e:
    print("MIMIR FAILED")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
