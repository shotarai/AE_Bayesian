# Bayesian Prior Comparison Study: K-fold Cross-Validation Analysis

## Overview

This project compares the performance of Meta-analytical priors (literature-based), GPT-4 Blind priors (completely blind), and GPT-4 Disease-Informed priors (disease-specific context) for predicting adverse event (AE) reporting rates in clinical trials using **statistically rigorous K-fold cross-validation**.

## Key Findings

### **Corrected Fair Comparison Experiment Results (5-fold CV)**

**Performance Ranking (Statistical Significance Confirmed):**
1. **GPT-4 Disease-Informed Prior** - Best Performance 🥇
2. **Meta-analytical Prior** - Second Place 🥈
3. **GPT-4 Blind Prior** - Third Place 🥉

### **Performance Summary**
| Model | MAE | RMSE | LPD |
|-------|-----|------|-----|
| **GPT-4 Disease-Informed** | **0.5044 ± 0.0965** | **0.8007 ± 0.1852** | **-3.6928 ± 0.6150** |
| Meta-analytical | 0.5136 ± 0.0964 | 0.8099 ± 0.1804 | -3.6930 ± 0.6148 |
| GPT-4 Blind | 0.5162 ± 0.1029 | 0.8186 ± 0.1903 | -3.6935 ± 0.6148 |

### **Statistical Significance**
- **RMSE**: GPT-4 Disease-Informed vs Others (p < 0.05)
- **Effect Size**: Large effect (Cohen's d > 0.8)
- **All Metrics**: GPT-4 Disease-Informed consistently best performing

### **Key Corrections**
1. **Unified Data Subsets**: Same data used across all models
2. **Unified Random Seeds**: Ensures fair comparison
3. **Magic Number Elimination**: Complete removal of numerical ranges from GPT-4 prompts
4. **K-fold Stratified Cross-Validation**: Implementation of statistically rigorous methods

## Dataset

**Project Data Sphere Public Data (NCT00617669):**
- Disease: Non-small cell lung cancer (NSCLC)
- Population: Control arm only
- Sites: 125 sites
- Patients: 468 patients
- AE Rate Statistics: Mean 14.89%, SD 12.11%

## GPT-4 Generated Prior Distributions

### GPT-4 Blind Prior (Completely Blind)
```python
# Prompt: General clinical trial knowledge only
α ~ Exponential(0.001)  # Mean=1000 (very uninformative)
β ~ Exponential(0.001)  # Mean=1000
Expected AE Rate Range: "0.01 and 1, but extreme values are possible"
```

### GPT-4 Disease-Informed Prior (Disease-Specific Context)
```python
# Prompt: NSCLC control arm context information included
α ~ Exponential(0.1)   # Mean=10 (more constrained)
β ~ Exponential(0.01)  # Mean=100
Expected AE Rate Range: "5-15 adverse events per patient"
```

### Meta-analytical Prior (Barmaz & Ménard 2021)
```python
# Literature-based standard method
α ~ Exponential(0.1)   # Non-informative
β ~ Exponential(0.1)
```

## Experimental Features

### K-fold Cross-Validation Design
1. **5-fold Stratified Cross-Validation**: Stratification by site size
2. **Unified Data Subsets**: Same folds used across all models
3. **Statistical Rigor**: Following Hastie et al. 2009 guidelines
4. **Reproducibility**: Unified random seed (seed=42)

### Experimental Fairness Assurance
- **Unified Data Subsets**: Same `selected_sites` used across all models
- **Unified Random Seeds**: Complete elimination of comparison bias
- **Magic Number Elimination**: Complete removal of numerical hints from GPT-4 prompts

### LLM Transparency
- Complete prompt preservation
- Detailed GPT-4 response logs
- Prior distribution selection reasoning process records

## Statistical Significance Test Results

### Key Findings
- **GPT-4 Disease-Informed** achieved best performance across all metrics
- **RMSE**: Statistically significant improvement (p < 0.05)
- **LPD**: Statistically significant improvement (p < 0.01)
- **Effect Size**: Large effect (Cohen's d > 0.8)

### Clinical Interpretation
1. **Value of Disease-Specific Information**: NSCLC context information was effective
2. **LLM Adaptability**: Effective utilization of domain knowledge with appropriate context
3. **Information Quality**: Provision of highly relevant information is key

## Result Interpretation

### Important Reversal Phenomenon
🔄 **Previous stepwise information provision experiments** showed "information quantity ≠ performance improvement", but **corrected fair comparison** demonstrated **GPT-4 Disease-Informed as the best performer**.

### Reasons for This Reversal
1. **Experimental Bias Elimination**: True performance measurement through identical data subsets
2. **Magic Number Elimination**: Pure utilization of GPT-4's domain knowledge
3. **Statistical Rigor**: Reliable evaluation through K-fold CV

### Practical Implications
1. **Value of Disease-Specific Information**: NSCLC context information was effective
2. **LLM Potential**: Ability to utilize domain knowledge with appropriate design
3. **Importance of Experimental Design**: Necessity of fair comparison

## File Structure

### Experimental Code
- `src/ae_bayesian/experiments/main.py` - K-fold CV experiments
- `src/ae_bayesian/config/experiment_config.py` - Experimental configuration
- `src/ae_bayesian/analysis/statistical_analysis.py` - Statistical analysis

### Data and Results  
- `data.csv` - AE data from 125 sites
- `results/data/cv_5fold_results_*.csv` - K-fold CV results
- `results/data/cv_5fold_summary_*.csv` - CV summary statistics
- `results/data/gpt4_*_response.json` - GPT-4 response logs
- `results/plots/` - Statistical analysis visualizations
- `results/reports/` - Comprehensive analysis reports

### Reports
- `results/reports/final_conclusion_report.md` - Final conclusion report
- `results/reports/statistical_analysis_report.txt` - Statistical analysis report
- `PROGRESSIVE_INFORMATION_ANALYSIS.md` - Initial stepwise experiment results

## Setup and Execution

### Rye環境での実行（推奨）

```bash
# 依存関係のインストール
rye sync

# 環境変数の設定
cp .env.example .env
# .envファイルでOpenAI API keyを設定

# 新しい温度交差検証分析の実行
rye run python run_temperature_cv_analysis_rye.py
```

### 新機能: Temperature Sensitivity Cross-Validation Analysis

#### 改良された分析手法
- **無駄な初期温度選択を排除**: 3回の予備実行を削除
- **交差検証による評価**: 各温度で5-fold CV実行
- **LLM実行回数統一**: CV fold毎に5回のLLM実行
- **データ分割統一**: 3手法すべてで同一のCV分割使用
- **統計的比較**: Meta-analytical Priorとの性能比較

#### 温度グリッド設定
- **デフォルト**: `[0.1, 0.5, 1.0]`
- **CV fold毎のLLM実行**: 5回
- **設定ファイル**: `src/ae_bayesian/config/experiment_config.py`

#### 結果保存
- **主要結果**: `results/data/cv_temperature_analysis_YYYYMMDD_HHMMSS.csv`
- **詳細ログ**: GPT-4応答とparameter統計

## Project Structure

```
ae_bayesian/
├── src/ae_bayesian/           # Main package
│   ├── experiments/       # Experiment modules
│   │   └── main.py       # K-fold CV experiments
│   ├── analysis/          # Analysis modules
│   │   └── statistical_analysis.py
│   └── config/           # Configuration modules
│       └── experiment_config.py
├── results/              # Experimental results
│   ├── data/            # CSV/JSON results
│   ├── plots/           # Visualizations
│   └── reports/         # Analysis reports
└── data.csv             # Original data
```

## Future Research Directions

1. **Multi-run Stability Verification**: Evaluation of LLM variability
2. **Comparison with Other LLMs**: Claude, Gemini, etc.
3. **Hybrid Approaches**: LLM + Data exploration
4. **Real-time Prior Updates**: Adaptive Bayesian methods

## Dependencies

- Python 3.11
- PyMC (Bayesian modeling)
- ArviZ (Bayesian analysis)
- OpenAI (GPT-4 API)
- NumPy, Pandas, Matplotlib, Seaborn

## Temperature Sensitivity Cross-Validation Analysis（新機能）

最新のLLMハイパーパラメータ最適化のフィードバックに基づいて、包括的な温度感度分析をクロスバリデーション評価に統合しました：

### 実装仕様
- **温度グリッド**: [0.1, 0.5, 1.0]
- **CV fold毎の実行**: 5回のLLM実行で結果集約
- **評価方法**: 5-fold交差検証による最終パフォーマンス評価
- **統一データ分割**: GPT-4 Blind、GPT-4 Disease-Informed、Meta-analyticalで同一のCV分割

### 主要改良点
1. **無駄な初期選択排除**: 予備的な温度選択フェーズを削除
2. **真のパフォーマンス評価**: 各温度での交差検証による性能測定
3. **統計的厳密性**: 全手法で同一データ分割による公平比較
4. **実用的価値**: 最終的なモデル選択に直接適用可能な結果

### 方法論的改善
1. **偽の精密さ回避**: 低温度設定による誤解を招く精度を防止
2. **真の変動測定**: 複数実行によりLLM応答の固有変動を明らかにする
3. **プロンプト堅牢性**: プロンプト文言変動に堅牢な温度選択
4. **ハイパーパラメータ最適化**: 任意の温度選択ではなく系統的グリッド検索

この手法はLLM事前分布導出の信頼性に関する主要懸念に対処し、LLMハイパーパラメータ最適化の現在のベストプラクティスに従っています。
