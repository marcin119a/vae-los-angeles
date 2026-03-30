#!/bin/bash

# Multi-Modal VAE Pipeline Runner

set -e  # Exit on error

echo "========================================"
echo "Multi-Modal VAE Training Pipeline"
echo "========================================"

# Step 1: Data Preparation
echo ""
echo "Step 1/3: Preparing data..."
python scripts/prepare_data.py

# Step 2: Training
echo ""
echo "Step 2/3: Training models..."
python train_dna2rna.py
python train_rna2dna.py

# Step 3: Evaluation
echo ""
echo "Step 3/3: Evaluating and comparing models..."
python evaluate_test.py
python scripts/compare_learning_curves.py
python scripts/compare_mse.py

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Model checkpoints: checkpoints/best_*.pt"
echo "  - Comparative plots: plots/comparisons/"
echo "  - Loss history: plots/loss_history_*.json"
echo "  - Evaluation results: plots/eval_results_*.json"
echo ""

