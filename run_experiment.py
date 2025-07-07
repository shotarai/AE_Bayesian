#!/usr/bin/env python3
"""
Improved Bayesian Prior Comparison Experiment Runner Script
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ae_bayesian.experiments.main import ImprovedBayesianExperiment

def main():
    """Main execution function"""
    print("=" * 60)
    print("Improved Bayesian Prior Comparison Experiment")
    print("=" * 60)
    
    # Create experiment instance
    experiment = ImprovedBayesianExperiment()
    
    # Execute experiment
    results = experiment.run_baseline_comparison()
    
    print("\nExperiment completed! Results saved in results directory.")
    
    return results

if __name__ == "__main__":
    main()
