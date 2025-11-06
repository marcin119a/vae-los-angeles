# Multi-Modal VAE for Cancer Genomics

A PyTorch implementation of Multi-Modal Variational Autoencoders (VAE and CVAE) for analyzing cancer genomics data. This project includes both standard VAE and Conditional VAE (CVAE) implementations that jointly learn representations from RNA expression, DNA methylation, and primary site information.

## 🧬 Overview

This project implements two model architectures:

### 1. **Multi-Modal VAE** (Standard)
A β-VAE with three modalities:
- **Modality A**: RNA expression data (TPM values)
- **Modality B**: DNA methylation data (beta values)
- **Modality C**: Primary tumor site labels

### 2. **Conditional Multi-Modal VAE (CVAE)** ⭐ NEW
An extension of VAE that conditions generation on primary site labels:
- **Conditional encoding**: Encoders receive data + site labels
- **Conditional decoding**: Decoders receive latent vector + site labels
- **Controlled generation**: Generate data for specific cancer types
- **Better separation**: Improved class separation in latent space

Both models can perform cross-modal reconstruction, enabling prediction of one modality from another (e.g., predicting RNA expression from DNA methylation patterns).

## 📁 Project Structure

```
vae-los-angeles/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py      # Encoder architectures for each modality
│   │   ├── decoders.py      # Decoder architectures for each modality
│   │   └── vae.py           # Main MultiModalVAE model
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py       # Dataset and DataLoader utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   └── losses.py        # Loss functions
│   └── config.py            # Configuration parameters
├── scripts/
│   └── prepare_data.py      # Data download and preprocessing
├── checkpoints/             # Saved model checkpoints
├── plots/                   # Generated plots and visualizations
├── data/                    # Processed data files
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vae-los-angeles
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

Download and prepare the datasets from Kaggle:

```bash
python scripts/prepare_data.py
```

This will:
- Download RNA expression and mutations data
- Download DNA methylation data
- Process and merge all datasets
- Normalize RNA expression values
- Encode primary site labels
- Save processed data to `data/`

### Training

Train the Multi-Modal VAE:

```bash
python train.py
```

Training features:
- β-VAE with warmup schedule
- Early stopping with patience
- Learning rate scheduling
- Automatic checkpointing of best model
- Training/validation loss visualization

### Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This will:
- Load the best model checkpoint
- Perform cross-modal reconstructions
- Compute evaluation metrics (MSE, MAE, Cosine Similarity, Pearson r)
- Generate visualizations
- Save results to `plots/`

## 📊 Model Architecture

### Encoders

- **EncoderA** (RNA): 782 → 128 → 20 (latent)
- **EncoderB** (DNA): 572 → 512 → 256 → 20 (latent)
- **EncoderC** (Site): Embedding(n_sites, 32) → 20 (latent)

### Decoders

- **DecoderA** (RNA): 20 → 128 → 782
- **DecoderB** (DNA): 20 → 256 → 512 → 572
- **DecoderC** (Site): 20 → 64 → n_sites

### Loss Function

The total loss combines:
- Reconstruction loss (MSE for RNA and DNA)
- Classification loss (Cross-entropy for primary site)
- KL divergence (regularization)

```
L = L_recon + γ·L_class + β·KL
```

## Configuration

Key hyperparameters can be modified in `src/config.py`:

- `LATENT_DIM`: Dimension of latent space (default: 20)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Initial learning rate (default: 5e-4)
- `NUM_EPOCHS`: Maximum training epochs (default: 200)
- `PATIENCE`: Early stopping patience (default: 15)
- `BETA_START`: KL divergence weight (default: 1e-3)
- `BETA_WARMUP_EPOCHS`: Epochs for β warmup (default: 50)

## Results

The model generates several outputs:

1. **Training curves**: Loss plots during training
2. **Reconstruction examples**: Visual comparison of original vs reconstructed data
3. **Correlation distributions**: Pearson correlation histograms
4. **Quantitative metrics**: JSON file with MSE, MAE, cosine similarity, and Pearson r

## Use Cases

- **Cross-modal prediction**: Predict RNA expression from DNA methylation
- **Missing data imputation**: Reconstruct missing modalities
- **Feature learning**: Extract joint representations of multi-omics data
- **Biomarker discovery**: Identify important features in latent space
