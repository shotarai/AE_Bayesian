"""
ae_exp2: Bayesian Prior Comparison Study

このパッケージは、ベイズ統計における事前分布の比較研究を行うためのツールです。
Meta-analytical priorとGPT-4生成priorの性能を比較し、
posterior predictive performanceとサンプルサイズの関係を解析します。
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__description__ = "Bayesian Prior Comparison Study: Meta-analytical vs GPT-4 Generated Priors"

# 主要モジュールのインポート
try:
    from .experiments.main import ImprovedBayesianExperiment
    from .analysis.improved_analysis import ImprovedPriorComparisonAnalysis
    
    __all__ = [
        "ImprovedBayesianExperiment", 
        "ImprovedPriorComparisonAnalysis",
    ]
except ImportError:
    # 開発時の依存関係エラーを回避
    __all__ = []

def hello() -> str:
    return "Hello from ae-exp2!"
