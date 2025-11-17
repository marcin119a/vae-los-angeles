
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory to save the plots
os.makedirs('plots', exist_ok=True)

# Load the processed data
data_path = 'data/processed_data.pkl'
df = pd.read_pickle(data_path)

# --- RNA Data ---

# Convert array data to matrix
rna_matrix = np.vstack(df['tpm_unstranded'].values)
n_genes = rna_matrix.shape[1]

# Calculate variance for each gene across samples
gene_variances = np.var(rna_matrix, axis=0)

# Select top 10 genes with the highest variance
top_genes_high_idx = np.argsort(gene_variances)[-10:]
rna_high_data = []
for idx in top_genes_high_idx:
    for sample_idx in range(len(df)):
        rna_high_data.append({
            'gene_id': f'Gene_{idx}',
            'tpm_unstranded': rna_matrix[sample_idx, idx]
        })
rna_df_plot_high = pd.DataFrame(rna_high_data)

# Create RNA boxplot for highest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='gene_id', y='tpm_unstranded', data=rna_df_plot_high, 
            order=[f'Gene_{idx}' for idx in top_genes_high_idx])
plt.title('RNA Transcription Across Samples (Top 10 Genes with Highest Variance)')
plt.ylabel('log1p(TPM)')
plt.xlabel('Gene')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/rna_genes_distribution_highest_variance.png')
plt.close()

# Select top 10 genes with the lowest variance (excluding zeros)
non_zero_var = gene_variances[gene_variances > 0]
if len(non_zero_var) >= 10:
    top_genes_low_idx = np.argsort(gene_variances)[np.where(gene_variances > 0)[0][:10]]
else:
    top_genes_low_idx = np.argsort(gene_variances)[:10]
    
rna_low_data = []
for idx in top_genes_low_idx:
    for sample_idx in range(len(df)):
        rna_low_data.append({
            'gene_id': f'Gene_{idx}',
            'tpm_unstranded': rna_matrix[sample_idx, idx]
        })
rna_df_plot_low = pd.DataFrame(rna_low_data)

# Create RNA boxplot for lowest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='gene_id', y='tpm_unstranded', data=rna_df_plot_low,
            order=[f'Gene_{idx}' for idx in top_genes_low_idx])
plt.title('RNA Transcription Across Samples (Top 10 Genes with Lowest Variance)')
plt.ylabel('log1p(TPM)')
plt.xlabel('Gene')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/rna_genes_distribution_lowest_variance.png')
plt.close()


# --- DNA Data ---

# Convert array data to matrix
dna_matrix = np.vstack([np.array(x) for x in df['beta_value'].values])
n_probes = dna_matrix.shape[1]

# Calculate variance for each probe across samples
probe_variances = np.var(dna_matrix, axis=0)

# Select top 10 probes with the highest variance
top_probes_high_idx = np.argsort(probe_variances)[-10:]
dna_high_data = []
for idx in top_probes_high_idx:
    for sample_idx in range(len(df)):
        dna_high_data.append({
            'probe_id': f'Probe_{idx}',
            'beta_value': dna_matrix[sample_idx, idx]
        })
dna_df_plot_high = pd.DataFrame(dna_high_data)

# Create DNA boxplot for highest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='probe_id', y='beta_value', data=dna_df_plot_high,
            order=[f'Probe_{idx}' for idx in top_probes_high_idx])
plt.title('DNA Methylation Across Samples (Top 10 CpG Islands with Highest Variance)')
plt.ylabel('Beta Value')
plt.xlabel('CpG Island')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/dna_cpg_distribution_highest_variance.png')
plt.close()

# Select top 10 probes with the lowest variance (excluding zeros)
non_zero_var = probe_variances[probe_variances > 0]
if len(non_zero_var) >= 10:
    top_probes_low_idx = np.argsort(probe_variances)[np.where(probe_variances > 0)[0][:10]]
else:
    top_probes_low_idx = np.argsort(probe_variances)[:10]

dna_low_data = []
for idx in top_probes_low_idx:
    for sample_idx in range(len(df)):
        dna_low_data.append({
            'probe_id': f'Probe_{idx}',
            'beta_value': dna_matrix[sample_idx, idx]
        })
dna_df_plot_low = pd.DataFrame(dna_low_data)

# Create DNA boxplot for lowest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='probe_id', y='beta_value', data=dna_df_plot_low,
            order=[f'Probe_{idx}' for idx in top_probes_low_idx])
plt.title('DNA Methylation Across Samples (Top 10 CpG Islands with Lowest Variance)')
plt.ylabel('Beta Value')
plt.xlabel('CpG Island')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/dna_cpg_distribution_lowest_variance.png')
plt.close()

print("Boxplots for highest and lowest variance genes/CpGs created and saved in the 'plots' directory.")
