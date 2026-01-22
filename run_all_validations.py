"""
Script to run all validations and comparisons for VAE models
Including CVAE, directional VAE, and cross-validation
"""
import os
import sys
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print("-"*80)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def main():
    """Run all validation scripts"""
    print("="*80)
    print("COMPREHENSIVE VALIDATION AND COMPARISON PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. Evaluate CVAE models (reconstruction quality)
    print("\n" + "="*80)
    print("STEP 1: CVAE EVALUATION (Reconstruction)")
    print("="*80)
    results['cvae'] = run_command(
        "python evaluate_cvae.py",
        "CVAE Evaluation"
    )
    
    # 2. Cross-validation comparison (includes CVAE)
    print("\n" + "="*80)
    print("STEP 2: CROSS-VALIDATION COMPARISON")
    print("="*80)
    print("This will compare all methods including CVAE, VAE, AE, kNN, and Mean baseline")
    print("Note: This may take a while depending on the number of folds and epochs...")
    
    # Ask user for parameters or use defaults
    use_defaults = input("\nUse default parameters? (y/n, default=y): ").strip().lower()
    
    if use_defaults == 'n':
        folds = input("Number of CV folds (default=10): ").strip() or "10"
        subset = input("Data subset fraction (default=0.1): ").strip() or "0.1"
        epochs = input("Number of epochs (default=200): ").strip() or "200"
        batch_size = input("Batch size (default=32): ").strip() or "32"
        neighbors = input("k values for kNN (default='5 10'): ").strip() or "5 10"
        
        cmd = f"python vae_cross_modality_cv.py --folds {folds} --subset {subset} --epochs {epochs} --batch_size {batch_size} --neighbors {neighbors}"
    else:
        cmd = "python vae_cross_modality_cv.py --folds 10 --subset 0.1 --epochs 200 --batch_size 32 --neighbors 5 10"
    
    results['cross_validation'] = run_command(
        cmd,
        "Cross-Validation Comparison"
    )
    
    # 3. Directional imputation comparison (if models are trained)
    print("\n" + "="*80)
    print("STEP 3: DIRECTIONAL IMPUTATION COMPARISON")
    print("="*80)
    print("This compares RNA2DNAVAE and DNA2RNAVAE with baseline methods")
    print("Note: Requires trained RNA2DNAVAE and DNA2RNAVAE models")
    
    run_directional = input("\nRun directional imputation comparison? (y/n, default=y): ").strip().lower()
    if run_directional != 'n':
        results['directional'] = run_command(
            "python compare_directional_imputation.py",
            "Directional Imputation Comparison"
        )
    else:
        results['directional'] = None
        print("Skipping directional imputation comparison")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION PIPELINE SUMMARY")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED" if success is False else "⊘ SKIPPED"
        print(f"  {step:20s}: {status}")
    
    print("\n" + "="*80)
    print("All validation scripts have been executed.")
    print("Check the 'plots/' directory for results and visualizations.")
    print("="*80)


if __name__ == "__main__":
    main()
