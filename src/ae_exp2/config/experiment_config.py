"""
Experimental Configuration File - Magic Number Elimination
"""

import os
from pathlib import Path

# Directory configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
REPORTS_DIR = RESULTS_DIR / "reports"
CONFIG_DIR = PROJECT_ROOT / "src" / "ae_exp2" / "config"

# Create directories
for dir_path in [RESULTS_DIR, PLOTS_DIR, DATA_DIR, REPORTS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "meta_analytical": {
        "name": "Meta-analytical Prior",
        "alpha_prior": {"distribution": "exponential", "parameters": [0.1]},
        "beta_prior": {"distribution": "exponential", "parameters": [0.1]},
        "description": "Barmaz & Ménard (2021) non-informative priors"
    },
    "gpt4_fallback": {
        "name": "GPT-4 Fallback Prior", 
        "alpha_prior": {"distribution": "exponential", "parameters": [0.1]},
        "beta_prior": {"distribution": "exponential", "parameters": [0.1]},
        "description": "Conservative fallback - same as meta-analytical"
    }
}

# Experimental parameters
EXPERIMENT_CONFIG = {
    "sample_sizes": [10, 20, 30, 50, 75, 100],
    "mcmc_samples": 1000,
    "mcmc_tune": 1000,
    "random_seed": 42,
    "n_experiments": 5  # Multiple runs for stability verification
}

# GPT-4 configuration
GPT4_CONFIG = {
    "model": "gpt-4",
    "max_tokens": 1500,
    "temperature": 0.1,  # Set low for more consistent responses
    "max_retries": 3,
    "timeout": 30
}

# Performance evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["mae", "rmse", "lpd"],
    "baseline_performance_levels": [0.8, 1.0, 1.2, 1.5],  # Baseline performance levels
    "significance_level": 0.05
}

# Visualization configuration
PLOT_CONFIG = {
    "style": "seaborn-v0_8",
    "figsize": (12, 8),
    "dpi": 300,
    "format": "png"
}
