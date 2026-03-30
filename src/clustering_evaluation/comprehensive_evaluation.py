
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import subprocess
from datetime import datetime
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Global styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
METHOD_COLORS = {
    'Mean': '#8da0cb',
    'KNN': '#fc8d62',
    'VAE': '#a6d854',
    'MIMIR': '#ffd92f',
    'CpGPT': '#66c2a5'
}

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
    "Cervix uteri": "Cervix",
    "Esophagus": "Esophagus",
    "Adrenal gland": "Adrenal",
    "Other and unspecified parts of tongue": "Tongue",
    "Connective, subcutaneous and other soft tissues": "Connective"
}

def main():
    print("="*80 + "\nEXTENSIVE CROSS-MODEL EVALUATION PIPELINE\n" + "="*80)
    
    train_df = pd.read_pickle('data/processed_data.pkl')
    train_subset, test_subset = train_test_split(train_df, test_size=0.1, random_state=42)
    test_subset.to_pickle('data/temp_test_subset.pkl')
    
    true_dna = np.stack(test_subset['beta_value'].values).astype(np.float32)
    true_rna = np.stack(test_subset['tpm_unstranded'].values).astype(np.float32)
    with open('data/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    labels = le.transform(test_subset['primary_site'])
    
    results = []
    clust_res = []
    plot_dir = 'plots/clustering_comprehensive'
    os.makedirs(plot_dir, exist_ok=True)

    def eval_model(name, target, pred_vals):
        mse = mean_squared_error(true_dna if target=='DNA' else true_rna, pred_vals)
        r, _ = pearsonr((true_dna if target=='DNA' else true_rna).flatten(), pred_vals.flatten())
        results.append({'Method': name, 'Target': target, 'MSE': float(mse), 'Pearson r': float(r)})
        
        if target == 'DNA':
            feats = np.hstack((true_rna, pred_vals))
            sc = StandardScaler().fit_transform(feats)
            ts = TSNE(n_components=2, random_state=42).fit_transform(sc)
            sil = float(silhouette_score(ts, labels))
            clust_res.append({'Method': name, 'Silhouette': sil})
            plt.figure(figsize=(8,6)); plt.scatter(ts[:,0], ts[:,1], c=labels, cmap='tab20', s=15, alpha=0.6)
            plt.title(f"{name} (Sil: {sil:.3f})"); plt.savefig(f"{plot_dir}/tsne_{name.lower()}.png"); plt.close()

    # 1. Mean
    dna_mean = np.mean(np.stack(train_subset['beta_value'].values), axis=0)
    rna_mean = np.mean(np.stack(train_subset['tpm_unstranded'].values), axis=0)
    eval_model('Mean', 'DNA', np.tile(dna_mean, (len(test_subset), 1)))
    eval_model('Mean', 'RNA', np.tile(rna_mean, (len(test_subset), 1)))

    # 2. KNN
    knn_dna = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(np.stack(train_subset['tpm_unstranded'].values), np.stack(train_subset['beta_value'].values))
    eval_model('KNN', 'DNA', knn_dna.predict(true_rna))
    knn_rna = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(np.stack(train_subset['beta_value'].values), np.stack(train_subset['tpm_unstranded'].values))
    eval_model('KNN', 'RNA', knn_rna.predict(true_dna))

    # 3. VAE (Subprocess)
    for t in ['DNA', 'RNA']:
        mod = 'rna2dna' if t=='DNA' else 'dna2rna'
        v_class = 'RNA2DNAVAE' if t=='DNA' else 'DNA2RNAVAE'
        in_col = 'tpm_unstranded' if t=='DNA' else 'beta_value'
        script = f"""
import sys, os, torch, pandas as pd, numpy as np, json, pickle
from scipy.stats import pearsonr
sys.path.append(os.path.abspath('.'))
from src.models import {v_class}
from src.config import Config
test = pd.read_pickle('data/temp_test_subset.pkl')
with open('data/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
device = torch.device('cpu')
model = {v_class}(Config.INPUT_DIM_A, Config.INPUT_DIM_B, len(le.classes_), Config.LATENT_DIM).to(device)
run_id = open('latest_{mod}_run_id.txt').read().strip()
model.load_state_dict(torch.load(f'checkpoints/best_{mod}_{{run_id}}.pt', map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    in_data = torch.tensor(np.stack(test['{in_col}'].values)).float()
    site = torch.tensor(le.transform(test['primary_site'])).long()
    if '{t}' == 'DNA':
        outs = model(rna=in_data, site=site)
    else:
        outs = model(dna=in_data, site=site)
    preds = outs[0].numpy()
with open('data/temp_res_vae_{t.lower()}.json', 'wb') as f: pickle.dump(preds, f)
"""
        script_path = f'temp_vae_{t}.py'
        with open(script_path, 'w') as f: f.write(script)
        subprocess.run(['venv/bin/python3', script_path])
        if os.path.exists(f'data/temp_res_vae_{t.lower()}.json'):
            preds = pickle.load(open(f'data/temp_res_vae_{t.lower()}.json', 'rb'))
            eval_model('VAE', t, preds)
            os.remove(f'data/temp_res_vae_{t.lower()}.json')
        os.remove(script_path)

    # 4. MIMIR (Subprocess)
    for t in ['DNA', 'RNA']:
        script = f"""
import sys, os, torch, pandas as pd, numpy as np, json, pickle
sys.path.insert(0, os.path.abspath('MIMIR'))
from src.mae_masked import MultiModalWithSharedSpace, load_modality_with_config, extract_encoder_decoder_from_pretrained
test = pd.read_pickle('data/temp_test_subset.pkl')
device = torch.device('cpu')
ae_rna, _, _ = load_modality_with_config('MIMIR/mimir_checkpoints/rna_ae.pt', map_location=device)
ae_dna, _, _ = load_modality_with_config('MIMIR/mimir_checkpoints/dna_ae.pt', map_location=device)
e_r, d_r = extract_encoder_decoder_from_pretrained(ae_rna); e_d, d_d = extract_encoder_decoder_from_pretrained(ae_dna)
sm = MultiModalWithSharedSpace({{'rna':e_r, 'dna':e_d}}, {{'rna':d_r, 'dna':d_d}}, {{'rna':256, 'dna':256}}, 128, 1).to(device)
sm.load_state_dict(torch.load('MIMIR/mimir_checkpoints/finetuned/shared_model_ep100.pt', map_location=device, weights_only=True))
sm.eval()
with torch.no_grad():
    in_mod, out_mod = ('rna', 'dna') if '{t}'=='DNA' else ('dna', 'rna')
    feat = torch.tensor(np.stack(test['tpm_unstranded' if in_mod=='rna' else 'beta_value'].values)).float()
    shared = sm.projections[in_mod](sm.encoders[in_mod](feat))
    preds = sm.decoders[out_mod](sm.rev_projections[out_mod](shared)).numpy()
with open('data/temp_res_mimir_{t.lower()}.json', 'wb') as f: pickle.dump(preds, f)
"""
        script_path = f'temp_mimir_{t}.py'
        with open(script_path, 'w') as f: f.write(script)
        subprocess.run(['venv/bin/python3', script_path])
        if os.path.exists(f'data/temp_res_mimir_{t.lower()}.json'):
            preds = pickle.load(open(f'data/temp_res_mimir_{t.lower()}.json', 'rb'))
            eval_model('MIMIR', t, preds)
            os.remove(f'data/temp_res_mimir_{t.lower()}.json')
        os.remove(script_path)

    # 5. CpGPT
    if os.path.exists('data/test_subset_cpgpt.pkl'):
        cpgpt_data = pd.read_pickle('data/test_subset_cpgpt.pkl')
        eval_model('CpGPT', 'DNA', np.stack(cpgpt_data['imputed_beta_value'].values))

    # --- FINAL SUMMARY ---
    acc_df = pd.DataFrame(results)
    clust_df = pd.DataFrame(clust_res)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(data=acc_df, x='Target', y='Pearson r', hue='Method', palette=METHOD_COLORS, ax=axes[0])
    axes[0].set_title("Reconstruction Accuracy (Pearson Correlation)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.barplot(data=clust_df, x='Method', y='Silhouette', palette=METHOD_COLORS, ax=axes[1])
    axes[1].set_title("Biological Structure Preservation (Silhouette Score)")
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/comprehensive_comparison_{ts}.png")
    print(f"\nSUCCESS! Extensive comparison complete.")
    print(f"Master plot saved to: {plot_dir}/comprehensive_comparison_{ts}.png")
    
    if os.path.exists('data/temp_test_subset.pkl'): os.remove('data/temp_test_subset.pkl')

if __name__ == "__main__":
    main()
