"""
設定モジュール

実験パラメータと設定を管理します。
"""

from .experiment_config import (
    MODEL_CONFIG,
    EXPERIMENT_CONFIG,
    GPT4_CONFIG,
    EVALUATION_CONFIG,
    PLOT_CONFIG,
    RESULTS_DIR,
    PLOTS_DIR,
    DATA_DIR,
    REPORTS_DIR
)

__all__ = [
    "MODEL_CONFIG",
    "EXPERIMENT_CONFIG", 
    "GPT4_CONFIG",
    "EVALUATION_CONFIG",
    "PLOT_CONFIG",
    "RESULTS_DIR",
    "PLOTS_DIR",
    "DATA_DIR",
    "REPORTS_DIR"
]
