#!/usr/bin/env python3
"""
改良版ベイズ事前分布比較実験のランナースクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ae_exp2.experiments.main import ImprovedBayesianExperiment

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("改良版ベイズ事前分布比較実験")
    print("=" * 60)
    
    # 実験インスタンスを作成
    experiment = ImprovedBayesianExperiment()
    
    # 実験実行
    results = experiment.run_baseline_comparison()
    
    print("\n実験完了！結果は results ディレクトリに保存されました。")
    
    return results

if __name__ == "__main__":
    main()
