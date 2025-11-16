
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

# Explode the dataframe to long format for RNA
rna_df_long = df[['case_barcode', 'gene_name', 'tpm_unstranded']].explode(['gene_name', 'tpm_unstranded'])
rna_df_long['tpm_unstranded'] = pd.to_numeric(rna_df_long['tpm_unstranded'], errors='coerce')


# Filter genes with at least 50% of samples
min_samples = len(df) * 0.5
gene_counts = rna_df_long['gene_name'].value_counts()
genes_to_keep = gene_counts[gene_counts >= min_samples].index
rna_df_filtered = rna_df_long[rna_df_long['gene_name'].isin(genes_to_keep)]

# Select top 10 genes with the highest variance
gene_variances_high = rna_df_filtered.groupby('gene_name')['tpm_unstranded'].var().nlargest(10)
top_genes_high = gene_variances_high.index
rna_df_plot_high = rna_df_filtered[rna_df_filtered['gene_name'].isin(top_genes_high)]

# Create RNA boxplot for highest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='gene_name', y='tpm_unstranded', data=rna_df_plot_high, order=top_genes_high)
plt.title('RNA Transcription Across Samples (Top 10 Genes with Highest Variance)')
plt.ylabel('log1p(TPM)')
plt.xlabel('Gene')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/rna_genes_distribution_highest_variance.png')
plt.close()

# Select top 10 genes with the lowest variance
gene_variances_low = rna_df_filtered.groupby('gene_name')['tpm_unstranded'].var().nsmallest(10)
top_genes_low = gene_variances_low.index
rna_df_plot_low = rna_df_filtered[rna_df_filtered['gene_name'].isin(top_genes_low)]

# Create RNA boxplot for lowest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='gene_name', y='tpm_unstranded', data=rna_df_plot_low, order=top_genes_low)
plt.title('RNA Transcription Across Samples (Top 10 Genes with Lowest Variance)')
plt.ylabel('log1p(TPM)')
plt.xlabel('Gene')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/rna_genes_distribution_lowest_variance.png')
plt.close()


# --- DNA Data ---

# Explode the dataframe to long format for DNA
dna_df_long = df[['case_barcode', 'probe_id_id', 'beta_value']].explode(['probe_id_id', 'beta_value'])
dna_df_long['beta_value'] = pd.to_numeric(dna_df_long['beta_value'], errors='coerce')


# Filter probes with at least 50% of samples
probe_counts = dna_df_long['probe_id_id'].value_counts()
probes_to_keep = probe_counts[probe_counts >= min_samples].index
dna_df_filtered = dna_df_long[dna_df_long['probe_id_id'].isin(probes_to_keep)]

# Select top 10 probes with the highest variance
probe_variances_high = dna_df_filtered.groupby('probe_id_id')['beta_value'].var().nlargest(10)
top_probes_high = probe_variances_high.index
dna_df_plot_high = dna_df_filtered[dna_df_filtered['probe_id_id'].isin(top_probes_high)]

# Create DNA boxplot for highest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='probe_id_id', y='beta_value', data=dna_df_plot_high, order=top_probes_high)
plt.title('DNA Methylation Across Samples (Top 10 CpG Islands with Highest Variance)')
plt.ylabel('Beta Value')
plt.xlabel('CpG Island')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/dna_cpg_distribution_highest_variance.png')
plt.close()

# Select top 10 probes with the lowest variance
probe_variances_low = dna_df_filtered.groupby('probe_id_id')['beta_value'].var().nsmallest(10)
top_probes_low = probe_variances_low.index
dna_df_plot_low = dna_df_filtered[dna_df_filtered['probe_id_id'].isin(top_probes_low)]

# Create DNA boxplot for lowest variance
plt.figure(figsize=(15, 8))
sns.boxplot(x='probe_id_id', y='beta_value', data=dna_df_plot_low, order=top_probes_low)
plt.title('DNA Methylation Across Samples (Top 10 CpG Islands with Lowest Variance)')
plt.ylabel('Beta Value')
plt.xlabel('CpG Island')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/dna_cpg_distribution_lowest_variance.png')
plt.close()

print("Boxplots for highest and lowest variance genes/CpGs created and saved in the 'plots' directory.")
