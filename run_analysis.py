#!/usr/bin/env python3
"""
改良版分析のランナースクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ae_exp2.analysis.improved_analysis import ImprovedPriorComparisonAnalysis

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("改良版詳細分析")
    print("=" * 60)
    
    # 分析インスタンスを作成
    analyzer = ImprovedPriorComparisonAnalysis()
    
    # 分析実行
    summary_stats = analyzer.run_comprehensive_analysis()
    
    print("\n分析完了！結果は results ディレクトリに保存されました。")
    
    return summary_stats

if __name__ == "__main__":
    main()
