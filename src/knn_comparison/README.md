# KNN Comparison Experiment

This directory contains code to compare K-Nearest Neighbors (KNN) regression models against VAE models for cross-modality imputation (RNA->DNA and DNA->RNA).

## Contents

- `run_comparison.py`: Main script that:
    1.  Loads processed data.
    2.  Trains and optimizes KNN models (Grid Search on `n_neighbors`, `weights`, `metric`).
        -   **Base**: Uses only the source modality.
        -   **Conditioned**: Uses a stratified approach where a separate `KNeighborsRegressor` is trained for each primary site (implemented in `ConditionedKNeighborsRegressor`).
    3.  Loads pre-trained VAE models (`DNA2RNAVAE` and `RNA2DNAVAE`) using the latest run IDs.
    4.  Evaluates all models on the validation set.
    5.  Generates comparison boxplots (Matplotlib and Plotly).

## Usage

Run the comparison from the project root:

```bash
python3 -m src.knn_comparison.run_comparison
```

## Outputs

Plots are saved to `plots/comparison/`:
- **Boxplots (Error Distribution)**:
    - `boxplot_RNA_to_DNA.png` / `.html`
    - `boxplot_DNA_to_RNA.png` / `.html`
- **t-SNE Plots (Latent Space/Prediction Structure)**:
    - `tsne_rna2dna_knn_base.png` / `.html`
    - `tsne_rna2dna_knn_cond.png` / `.html`
    - `tsne_rna2dna_vae_cond.png` / `.html`
    - `tsne_dna2rna_knn_base.png` / `.html`
    - `tsne_dna2rna_knn_cond.png` / `.html`
    - `tsne_dna2rna_vae_cond.png` / `.html`
