import os
import pandas as pd
import kagglehub

dna_path = kagglehub.dataset_download('martininf1n1ty/dna-metylation-onkodb')
dataset_path = dna_path

# Dictionary to store the loaded parquet files
parquet_dataframes = {}

# Iterate through the files in the dataset path
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file == 'filtered_data.parquet':
            continue
        if file.endswith('.parquet'):
            filepath = os.path.join(root, file)
            try:
                # Load the parquet file into a pandas DataFrame
                df = pd.read_parquet(filepath)
                # Store the DataFrame in the dictionary with the filename as the key
                parquet_dataframes[file] = df
                print(f"Loaded {file} successfully.")
            except Exception as e:
                print(f"Error loading {file}: {e}")

combined_df = pd.concat(parquet_dataframes.values(), ignore_index=True)

print(f"Shape before processing: {combined_df.shape}")
print(f"Missing values before processing:\n{combined_df.isnull().sum()}")

# Remove duplicates based on case_barcode and probe_id
combined_df = combined_df.drop_duplicates(subset=['case_barcode', 'probe_id'], keep='first')
print(f"\nShape after removing duplicates: {combined_df.shape}")

# Get unique patients and probes
unique_patients = combined_df['case_barcode'].unique()
unique_probes = combined_df['probe_id'].unique()

print(f"Unique patients: {len(unique_patients)}")
print(f"Unique probe_ids: {len(unique_probes)}")

# Create complete index (every patient should have every probe)
from itertools import product
complete_index = pd.DataFrame(
    list(product(unique_patients, unique_probes)),
    columns=['case_barcode', 'probe_id']
)

print(f"\nComplete matrix shape (patient × probe): {complete_index.shape}")

# Merge with existing data to fill in missing probe_ids for each patient
combined_df = complete_index.merge(
    combined_df,
    on=['case_barcode', 'probe_id'],
    how='left'
)

print(f"Shape after creating complete matrix: {combined_df.shape}")
print(f"Missing beta_values: {combined_df['beta_value'].isnull().sum()}")

# Mean imputation for each patient (grouped by case_barcode)
combined_df['beta_value'] = combined_df.groupby('case_barcode')['beta_value'].transform(
    lambda x: x.fillna(x.mean())
)

print(f"\nMissing values after imputation:\n{combined_df.isnull().sum()}")
print(f"Final shape: {combined_df.shape}")


combined_df.to_parquet('../data/dna_methylation_imputed.parquet')