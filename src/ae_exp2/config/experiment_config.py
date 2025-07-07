"""
実験設定ファイル - マジックナンバーを排除
"""

import os
from pathlib import Path

# ディレクトリ設定
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"
REPORTS_DIR = RESULTS_DIR / "reports"
CONFIG_DIR = PROJECT_ROOT / "src" / "ae_exp2" / "config"

# ディレクトリ作成
for dir_path in [RESULTS_DIR, PLOTS_DIR, DATA_DIR, REPORTS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# モデル設定
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

# 実験パラメータ
EXPERIMENT_CONFIG = {
    "sample_sizes": [10, 20, 30, 50, 75, 100],
    "mcmc_samples": 1000,
    "mcmc_tune": 1000,
    "random_seed": 42,
    "n_experiments": 5  # 複数回実行で安定性確認
}

# GPT-4設定
GPT4_CONFIG = {
    "model": "gpt-4",
    "max_tokens": 1500,
    "temperature": 0.1,  # より一貫した回答のため低く設定
    "max_retries": 3,
    "timeout": 30
}

# 性能評価設定
EVALUATION_CONFIG = {
    "metrics": ["mae", "rmse", "lpd"],
    "baseline_performance_levels": [0.8, 1.0, 1.2, 1.5],  # ベースライン性能レベル
    "significance_level": 0.05
}

# 可視化設定
PLOT_CONFIG = {
    "style": "seaborn-v0_8",
    "figsize": (12, 8),
    "dpi": 300,
    "format": "png"
}
