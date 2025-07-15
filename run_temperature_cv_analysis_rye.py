#!/usr/bin/env python3
"""
Temperature Sensitivity Cross-Validation Analysis with Rye
========================================================

This script implements the redesigned temperature sensitivity analysis with cross-validation
evaluation using Rye environment management.

Features:
1. Removes wasteful initial 3-run temperature selection
2. Implements cross-validation performance evaluation for each temperature  
3. Runs 5 LLM executions per temperature during cross-validation
4. Aggregates accuracy results and compares with Meta-analytical Prior
5. Ensures identical CV data splits across all three methods

Usage:
    rye run python run_temperature_cv_analysis_rye.py
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ae_bayesian.experiments.main import ImprovedBayesianExperiment
from ae_bayesian.config.experiment_config import *

def main():
    print("="*80)
    print("TEMPERATURE SENSITIVITY CROSS-VALIDATION ANALYSIS (RYE)")
    print("="*80)
    print("Redesigned approach:")
    print("- No wasteful initial temperature selection")
    print("- Cross-validation performance evaluation for each temperature")
    print("- 5 LLM executions per CV fold per temperature")
    print("- Identical CV splits across all methods")
    print("- Statistical comparison with Meta-analytical baseline")
    print("="*80)
    
    try:
        # Initialize experiment
        experiment = ImprovedBayesianExperiment("data.csv")
        
        # Run the new temperature-based cross-validation analysis
        print(f"\nRunning {len(GPT4_CONFIG['temperature_grid'])} temperatures x 5-fold CV...")
        print(f"Temperature grid: {GPT4_CONFIG['temperature_grid']}")
        print(f"LLM runs per CV fold: {GPT4_CONFIG['n_runs_per_cv_fold']}")
        
        # Execute cross-validation with temperature analysis
        results_df = experiment.run_cv_experiment_with_temperature_analysis(n_folds=5)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Results saved to: {DATA_DIR}")
        print(f"Total experiments conducted: {len(results_df)} model-fold-temperature combinations")
        
        # Summary statistics
        print(f"\nExperiment summary:")
        print(f"- Temperatures tested: {sorted(results_df['temperature'].unique())}")
        print(f"- Models compared: {sorted(results_df['model'].unique())}")
        print(f"- CV folds: {sorted(results_df['fold'].unique())}")
        
        # Show best performers
        print(f"\nBest overall performers:")
        mae_summary = results_df.groupby(['model', 'temperature'])['mae'].mean()
        best_mae = mae_summary.min()
        best_config = mae_summary.idxmin()
        print(f"Best MAE: {best_config[0]} (T={best_config[1]:.1f}) = {best_mae:.4f}")
        
        rmse_summary = results_df.groupby(['model', 'temperature'])['rmse'].mean()
        best_rmse = rmse_summary.min()
        best_rmse_config = rmse_summary.idxmin()
        print(f"Best RMSE: {best_rmse_config[0]} (T={best_rmse_config[1]:.1f}) = {best_rmse:.4f}")
        
        lpd_summary = results_df.groupby(['model', 'temperature'])['lpd'].mean()
        best_lpd = lpd_summary.max()
        best_lpd_config = lpd_summary.idxmax()
        print(f"Best LPD: {best_lpd_config[0]} (T={best_lpd_config[1]:.1f}) = {best_lpd:.4f}")
        
        return results_df
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = main()
    if results is not None:
        print(f"\n{'='*50}")
        print("TEMPERATURE CV ANALYSIS COMPLETE")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*30}")
        print("ANALYSIS FAILED")
        print(f"{'='*30}")
        sys.exit(1)
