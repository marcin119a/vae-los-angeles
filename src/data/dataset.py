"""
Dataset classes for Multi-Modal VAE
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal cancer genomics data
    
    Contains:
    - RNA expression data (tpm_unstranded)
    - DNA methylation data (beta_value)
    - Primary site labels
    """
    def __init__(self, dataframe):
        """
        Args:
            dataframe: pandas DataFrame with columns:
                - tpm_unstranded: RNA expression values (list)
                - beta_value: DNA methylation values (list)
                - primary_site_encoded: encoded primary site labels (int)
        """
        self.dataframe = dataframe
        self.tpm_data = np.array(self.dataframe['tpm_unstranded'].tolist()).astype(np.float32)
        self.beta_data = np.array(self.dataframe['beta_value'].tolist()).astype(np.float32)
        self.primary_site = np.array(self.dataframe['primary_site_encoded']).astype(np.int64)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tpm = torch.tensor(self.tpm_data[idx])
        beta = torch.tensor(self.beta_data[idx])
        site = torch.tensor(self.primary_site[idx])
        return tpm, beta, site

    @classmethod
    def from_numpy(cls, tpm_data, beta_data, primary_site):
        """
        Creates a dataset from numpy arrays.
        """
        df = pd.DataFrame({
            'tpm_unstranded': list(tpm_data),
            'beta_value': list(beta_data),
            'primary_site_encoded': primary_site
        })
        return cls(df)

