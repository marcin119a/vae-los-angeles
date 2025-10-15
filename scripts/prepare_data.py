"""
Script to download and prepare data from Kaggle
"""
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import LabelEncoder


def download_datasets():
    """Download datasets from Kaggle"""
    print("Downloading RNA and mutations dataset...")
    rna_path = kagglehub.dataset_download('martininf1n1ty/rna-mutations-all-datasets')
    print(f"RNA dataset downloaded to: {rna_path}")
    
    print("\nDownloading DNA methylation dataset...")
    dna_path = kagglehub.dataset_download('martininf1n1ty/dna-metylation-onkodb')
    print(f"DNA methylation dataset downloaded to: {dna_path}")
    
    return rna_path, dna_path


def prepare_rna_data(rna_path):
    """Prepare RNA expression data"""
    print("\nPreparing RNA expression data...")
    df_expressions = pd.read_parquet(f'{rna_path}/expression (1).parquet')
    
    # Sort by gene_name before grouping
    df_expressions_sorted = df_expressions.sort_values(by='gene_name')
    
    # Group by case_barcode and aggregate tpm_unstranded into a list
    grouped_expressions_df = df_expressions_sorted.groupby('case_barcode')['tpm_unstranded'].apply(list).reset_index()
    
    # Filter to keep only rows where the list length is 782
    filtered_grouped_expressions_df = grouped_expressions_df[
        grouped_expressions_df['tpm_unstranded'].apply(len) == 782
    ]
    
    print(f"RNA data shape: {filtered_grouped_expressions_df.shape}")
    return filtered_grouped_expressions_df


def prepare_mutation_data(rna_path):
    """Extract primary site information from mutations data"""
    print("\nExtracting primary site information...")
    df_mutations = pd.read_parquet(f'{rna_path}/mutations (1).parquet')
    
    primary_site_df = df_mutations[['case_barcode', 'primary_site']].drop_duplicates(
        subset=['case_barcode'], keep='first'
    )
    
    print(f"Primary site data shape: {primary_site_df.shape}")
    return primary_site_df


def prepare_dna_methylation_data(dna_path):
    """Prepare DNA methylation data"""
    print("\nPreparing DNA methylation data...")
    df = pd.read_parquet(f'{dna_path}/filtered_data.parquet')
    
    # Sort by probe_id before grouping
    df_sorted = df.sort_values(by='probe_id')
    grouped_df = df_sorted.groupby('case_barcode')['beta_value'].apply(list).reset_index()
    
    print(f"DNA methylation data shape: {grouped_df.shape}")
    return grouped_df


def merge_and_normalize_data(rna_df, dna_df, primary_site_df):
    """Merge all datasets and normalize"""
    print("\nMerging datasets...")
    
    # Merge DNA methylation with primary site
    merged_grouped_df = pd.merge(dna_df, primary_site_df, on='case_barcode')
    
    # Merge RNA expression with DNA methylation
    merged_df = pd.merge(rna_df, merged_grouped_df, on='case_barcode')
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Normalize tpm_unstranded data
    print("\nNormalizing RNA expression data...")
    merged_df["tpm_unstranded"] = merged_df["tpm_unstranded"].apply(
        lambda x: (np.array(x) - np.mean(x)) / (np.std(x) + 1e-8)
    )
    
    # Encode primary site labels
    print("Encoding primary site labels...")
    label_encoder = LabelEncoder()
    merged_df['primary_site_encoded'] = label_encoder.fit_transform(merged_df['primary_site'])
    
    print("\nPrimary site encoding (first 5):")
    for cls, code in zip(label_encoder.classes_[:5], range(min(5, len(label_encoder.classes_)))):
        print(f"{cls} → {code}")
    
    print(f"\nTotal number of primary sites: {len(label_encoder.classes_)}")
    
    return merged_df, label_encoder


def main():
    """Main data preparation pipeline"""
    # Download datasets
    rna_path, dna_path = download_datasets()
    
    # Prepare individual datasets
    rna_df = prepare_rna_data(rna_path)
    primary_site_df = prepare_mutation_data(rna_path)
    dna_df = prepare_dna_methylation_data(dna_path)
    
    # Merge and normalize
    merged_df, label_encoder = merge_and_normalize_data(rna_df, dna_df, primary_site_df)
    
    # Save processed data
    print("\nSaving processed data...")
    os.makedirs('data', exist_ok=True)
    merged_df.to_pickle('data/processed_data.pkl')
    
    # Save label encoder
    import pickle
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\nData preparation complete!")
    print(f"Processed data saved to: data/processed_data.pkl")
    print(f"Label encoder saved to: data/label_encoder.pkl")


if __name__ == "__main__":
    main()

