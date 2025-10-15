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
echo "Step 2/3: Training model..."
python train.py

# Step 3: Evaluation
echo ""
echo "Step 3/3: Evaluating model..."
python evaluate.py

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Model checkpoint: checkpoints/best_multivae.pt"
echo "  - Training plots: plots/training_losses.png"
echo "  - Evaluation results: plots/evaluation_results.json"
echo "  - Reconstruction examples: plots/reconstruction_example_*.png"
echo ""

