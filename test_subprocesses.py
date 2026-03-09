import subprocess
import os
import traceback

def run_tests():
    print("Testing VAE script execution...")
    try:
        if not os.path.exists("temp_vae_eval.py"):
            print("temp_vae_eval.py is missing! That implies it was removed.")
    except Exception as e:
        pass
        
    # We can reconstruct temp_vae_eval.py from the source code in cluster_imputation_mimir.py
    import json
    import pandas as pd
    
    df = pd.read_pickle('data/processed_data.pkl')
    from sklearn.model_selection import train_test_split
    _, test_subset = train_test_split(df, test_size=0.1, random_state=42)
    test_subset.to_pickle('data/temp_test_subset.pkl')

    eval_vae_script = """
import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from scipy.stats import pearsonr

# We need to load their VAE model
sys.path.append(os.path.abspath('.'))
from src.models.multimodal_vae import MultiModalVAE
from src.config import Config

test_subset = pd.read_pickle('data/temp_test_subset.pkl')
true_dna_te = np.array(test_subset['beta_value'].tolist()).astype(np.float32)
true_rna_te = np.array(test_subset['tpm_unstranded'].tolist()).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
cfg = Config()

model = MultiModalVAE(
    rna_input_dim=len(cfg.selected_rna_features),
    dna_input_dim=len(cfg.selected_dna_features),
    hidden_dims=cfg.hidden_dims,
    latent_dim=cfg.latent_dim,
    dropout_rate=cfg.dropout_rate
).to(device)

model_path = cfg.best_model_path if hasattr(cfg, 'best_model_path') and os.path.exists(cfg.best_model_path) else 'models/checkpoints/best_model.pt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rna_t = torch.tensor(true_rna_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, logvar = model.encode_rna(rna_t)
        z = model.reparameterize(mu, logvar)
        pred_dna = model.decode_dna(z).cpu().numpy()

    t_flat = true_dna_te.flatten()
    p_flat = pred_dna.flatten()
    mse = float(np.mean((t_flat - p_flat)**2))
    pearson = float(pearsonr(t_flat, p_flat)[0])

    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'MSE': mse, 'Pearson r': pearson}, f)
else:
    with open('data/temp_vae_acc.json', 'w') as f:
        json.dump({'error': 'No best_model.pt found'}, f)
"""
    with open('temp_vae_eval.py', 'w') as f:
        f.write(eval_vae_script)
        
    try:
        res = subprocess.run(['venv/bin/python', 'temp_vae_eval.py'], capture_output=True, text=True, check=True)
        print("VAE SUCCESS")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("VAE FAILED")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

    eval_script = """
import sys
import pandas as pd
import numpy as np
import torch
import json
from scipy.stats import pearsonr
from src.mae_masked import MultiModalWithSharedSpace
from src.config import Config

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
        
    try:
        res = subprocess.run(['../venv/bin/python', 'temp_eval.py'], cwd='MIMIR', capture_output=True, text=True, check=True)
        print("MIMIR SUCCESS")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("MIMIR FAILED")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == '__main__':
    run_tests()
