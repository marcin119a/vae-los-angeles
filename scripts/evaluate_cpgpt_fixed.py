
import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('cpgpt_repo_tmp'))

# Monkeypatch pyarrow
try:
    import pyarrow as pa
    if not hasattr(pa, 'PyExtensionType'):
        pa.PyExtensionType = pa.ExtensionType
except ImportError:
    pass

from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
from cpgpt.model.cpgpt_module import m_to_beta

DEPENDENCIES_DIR = "cpgpt_repo_tmp/dependencies"
LLM_DEPENDENCIES_DIR = DEPENDENCIES_DIR + "/human"
DATA_DIR = "cpgpt_repo_tmp/data"
PROCESSED_DIR = "data/cpgpt_processed_te"

def main():
    with open('data/cpg_probes.txt', 'r') as f:
        probes = [line.strip() for line in f.readlines()]

    df = pd.read_pickle('data/processed_data.pkl')
    _, test_subset = train_test_split(df, test_size=0.1, random_state=42)
    true_vals = np.stack(test_subset['beta_value'].values).astype(np.float32)
    
    arrow_df = pd.DataFrame(true_vals, columns=probes)
    arrow_df['GSM_ID'] = [f"s_{i}" for i in range(len(arrow_df))]
    arrow_df.set_index('GSM_ID', inplace=True)
    os.makedirs('data/cpgpt_arrow_te', exist_ok=True)
    ARROW_PATH = "data/cpgpt_arrow_te/test.arrow"
    arrow_df.to_feather(ARROW_PATH)
    
    inferencer = CpGPTInferencer(dependencies_dir=DEPENDENCIES_DIR, data_dir=DATA_DIR)
    MODEL_NAME = "small"
    MODEL_CHECKPOINT_PATH = f"{DEPENDENCIES_DIR}/model/weights/{MODEL_NAME}.ckpt"
    MODEL_CONFIG_PATH = f"{DEPENDENCIES_DIR}/model/config/{MODEL_NAME}.yaml"
    
    config = inferencer.load_cpgpt_config(MODEL_CONFIG_PATH)
    model = inferencer.load_cpgpt_model(config, model_ckpt_path=MODEL_CHECKPOINT_PATH, strict_load=False)
    
    embedder = DNALLMEmbedder(dependencies_dir=LLM_DEPENDENCIES_DIR)
    prober = IlluminaMethylationProber(dependencies_dir=LLM_DEPENDENCIES_DIR, embedder=embedder)
    
    datasaver = CpGPTDataSaver(data_paths=ARROW_PATH, processed_dir=PROCESSED_DIR)
    datasaver.process_files(prober, embedder)
    
    datamodule = CpGPTDataModule(
        predict_dir=PROCESSED_DIR,
        dependencies_dir=LLM_DEPENDENCIES_DIR,
        batch_size=1,
        num_workers=0
    )
    
    genomic_locations = prober.locate_probes(probes, "homo_sapiens")
    trainer = CpGPTTrainer(precision="16-mixed", accelerator='cpu')
    
    preds = trainer.predict(model=model, datamodule=datamodule, predict_mode="reconstruct",
                            genomic_locations=genomic_locations, species="homo_sapiens", return_keys=["pred_meth"])
    
    if isinstance(preds, list):
        if isinstance(preds[0], dict):
            pred_tensor = torch.cat([batch["pred_meth"] for batch in preds], dim=0)
        else:
            pred_tensor = torch.cat(preds, dim=0)
    else:
        pred_tensor = preds["pred_meth"]

    # FIX: Convert to float32 before m_to_beta
    pred_tensor = pred_tensor.to(torch.float32)
    predicted_beta = m_to_beta(pred_tensor).cpu().numpy()
    
    mse = np.mean((true_vals.flatten() - predicted_beta.flatten())**2)
    r, _ = pearsonr(true_vals.flatten(), predicted_beta.flatten())
    
    res = {'Method': 'CpGPT', 'Target': 'DNA', 'MSE': float(mse), 'Pearson r': float(r)}
    os.makedirs('plots', exist_ok=True)
    with open('plots/eval_results_CpGPT_cpgpt.json', 'w') as f:
        json.dump(res, f)
    
    test_subset['imputed_beta_value'] = list(predicted_beta)
    test_subset.to_pickle('data/test_subset_cpgpt.pkl')
    
    print(f"✓ Saved CpGPT results: MSE={mse:.4f}, r={r:.4f}")

if __name__ == '__main__':
    main()
