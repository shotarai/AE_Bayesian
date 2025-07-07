# Bayesian Prior Comparison Study: Meta-Analytical vs GPT-4 Generated Priors

## 概要

このプロジェクトは、臨床試験における有害事象（AE）報告率の予測において、従来のメタ分析的事前分布（Model A）とGPT-4生成事前分布（Model B）の効果を比較する研究です。目的は、同じ精度を達成するためにより小さなサンプルサイズで済む事前分布を特定することです。

## 主要な発見

- **GPT-4事前分布が一貫して優秀な性能を示す**
- サンプルサイズ10：MAE 23.5%改善、RMSE 20.7%改善
- サンプルサイズ20：MAE 24.5%改善、RMSE 22.3%改善
- サンプルサイズ30：MAE 9.5%改善、RMSE 12.7%改善
- 効果サイズ：Cohen's d = 2.08（非常に大きな効果）

## データ

125の臨床試験サイトのAE報告データを使用：
- 平均AE率：14.89%
- 標準偏差：12.11%
- 範囲：0.00% - 98.50%

## モデル

### Model A: メタ分析的事前分布
```python
α_study ~ Exponential(0.1)  # 非情報的分布
β_study ~ Exponential(0.1)
```

### Model B: GPT-4生成事前分布
```python
α_study ~ Gamma(1, 1)  # 情報的分布
β_study ~ Gamma(1, 1)
```

## ファイル構成

- `main.py` - 完全な実験フレームワーク
- `simple_experiment.py` - 簡略化された比較研究
- `detailed_analysis.py` - 詳細統計分析
- `data.csv` - 125サイトのAEデータ
- `RESEARCH_REPORT.md` - 包括的研究文書
- `PRESENTATION_SUMMARY.md` - 実行要約

## セットアップ

```bash
# 依存関係のインストール
rye sync

# 環境変数の設定
cp .env.example .env
# OpenAI API keyを.envに設定

# 実験の実行
rye run python run_experiment.py

# 分析の実行  
rye run python run_analysis.py
```

## プロジェクト構造

```
ae_exp2/
├── src/ae_exp2/           # メインパッケージ
│   ├── experiments/       # 実験モジュール
│   │   ├── improved_experiment.py
│   │   ├── main.py
│   │   └── simple_experiment.py
│   ├── analysis/          # 分析モジュール
│   │   └── improved_analysis.py
│   └── config/           # 設定モジュール
│       └── experiment_config.py
├── results/              # 実験結果
│   ├── data/            # 結果データ
│   ├── plots/           # 可視化
│   └── reports/         # レポート
├── run_experiment.py    # 実験実行スクリプト
├── run_analysis.py      # 分析実行スクリプト
└── data.csv            # 元データ
```

## 結果

実験結果は以下のファイルに保存されます：
- `simple_comparison_results.csv` - 実験データ
- `simple_comparison_results.png` - 基本比較可視化
- `prior_comparison.png` - 事前分布比較
- `detailed_comparison_analysis.png` - 詳細分析可視化

## 依存関係

- Python 3.11
- PyMC (ベイズモデリング)
- ArviZ (ベイズ分析)
- OpenAI (GPT-4 API)
- NumPy, Pandas, Matplotlib, Seaborn
