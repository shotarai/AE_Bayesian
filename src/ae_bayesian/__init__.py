"""
ae_bayesian: Bayesian Prior Comparison Study

This package provides tools for conducting comparative studies of prior distributions in Bayesian statistics.
It compares the performance of Meta-analytical priors and GPT-4 generated priors,
analyzing the relationship between posterior predictive performance and sample size.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__description__ = "Bayesian Prior Comparison Study: Meta-analytical vs GPT-4 Generated Priors"

# Import main modules
try:
    from .experiments.main import ImprovedBayesianExperiment
    from .analysis.improved_analysis import ImprovedPriorComparisonAnalysis
    
    __all__ = [
        "ImprovedBayesianExperiment", 
        "ImprovedPriorComparisonAnalysis",
    ]
except ImportError:
    # Avoid dependency errors during development
    __all__ = []

def hello() -> str:
    return "Hello from ae-exp2!"
