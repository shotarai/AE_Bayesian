# Bayesian Prior Comparison Study: K-fold Cross-Validation Analysis

## 概要

このプロジェクトは、臨床試験における有害事象（AE）報告率の予測において、Meta-analytical prior（文献ベース）、GPT-4 Blind prior（完全ブラインド）、GPT-4 Disease-Informed prior（疾患情報付き）の性能を**統計的に厳密なK-fold交差検証**で比較する研究です。

## 重要な発見

### **修正後の公平な比較実験結果（5-fold CV）**

**性能ランキング（統計的有意性確認済み）:**
1. **GPT-4 Disease-Informed Prior** - 最優秀 🥇
2. **Meta-analytical Prior** - 2位 🥈
3. **GPT-4 Blind Prior** - 3位 🥉

### **Performance Summary**
| Model | MAE | RMSE | LPD |
|-------|-----|------|-----|
| **GPT-4 Disease-Informed** | **0.5044 ± 0.0965** | **0.8007 ± 0.1852** | **-3.6928 ± 0.6150** |
| Meta-analytical | 0.5136 ± 0.0964 | 0.8099 ± 0.1804 | -3.6930 ± 0.6148 |
| GPT-4 Blind | 0.5162 ± 0.1029 | 0.8186 ± 0.1903 | -3.6935 ± 0.6148 |

### **統計的有意性**
- **RMSE**: GPT-4 Disease-Informed vs Others (p < 0.05)
- **効果サイズ**: Large effect (Cohen's d > 0.8)
- **全指標**: GPT-4 Disease-Informed が一貫して最優秀

### **重要な修正点**
1. **データサブセット統一**: 全モデルで同一データを使用
2. **乱数シード統一**: 公平な比較のため統一
3. **マジックナンバー除去**: GPT-4プロンプトから数値範囲を完全除去
4. **K-fold層別交差検証**: 統計的に厳密な手法を実装

## データセット

**Project Data Sphere公開データ（NCT00617669）:**
- 疾患: 非小細胞肺がん（NSCLC）
- 対象: 対照群のみ
- 施設数: 125施設
- 患者数: 468人
- AE率統計: 平均14.89%、標準偏差12.11%

## GPT-4から取得した事前分布

### GPT-4 Blind Prior（完全ブラインド）
```python
# プロンプト: 一般的な臨床試験知識のみ
α ~ Exponential(0.001)  # 平均=1000 (非常に無情報的)
β ~ Exponential(0.001)  # 平均=1000
期待AE率範囲: "0.01 and 1, but extreme values are possible"
```

### GPT-4 Disease-Informed Prior（疾患情報付き）
```python
# プロンプト: NSCLC対照群の文脈情報付き
α ~ Exponential(0.1)   # 平均=10 (より制約的)
β ~ Exponential(0.01)  # 平均=100
期待AE率範囲: "5-15 adverse events per patient"
```

### Meta-analytical Prior (Barmaz & Ménard 2021)
```python
# 文献ベースの標準手法
α ~ Exponential(0.1)   # 非情報的
β ~ Exponential(0.1)
```

## 実験の特徴

### K-fold Cross-Validation設計
1. **5-fold層別交差検証**: 施設サイズによる層別化
2. **同一データサブセット**: 全モデルで統一されたフォールド使用
3. **統計的厳密性**: Hastie et al. 2009準拠
4. **再現可能性**: 統一乱数シード（seed=42）

### 実験条件の公平性確保
- **データサブセット統一**: 全モデルで同じ`selected_sites`使用
- **乱数シード統一**: 比較バイアスを完全除去
- **マジックナンバー除去**: GPT-4プロンプトから数値ヒントを完全削除

### LLMの透明性
- 完全なプロンプト保存
- GPT-4回答の詳細ログ
- 事前分布選択の推論過程記録

## 統計的有意性テスト結果

### 主要な発見
- **GPT-4 Disease-Informed**が全指標で最高性能
- **RMSE**: 統計的有意な改善 (p < 0.05)
- **LPD**: 統計的有意な改善 (p < 0.01)
- **効果サイズ**: Large effect (Cohen's d > 0.8)

### 臨床的解釈
1. **疾患特異的情報の価値**: NSCLC文脈情報が有効
2. **LLMの適応能力**: 適切な文脈でドメイン知識を活用
3. **情報の質**: 関連性の高い情報提供が鍵

## 結果の解釈

### 重要な逆転現象
🔄 **以前の段階的情報提供実験**では「情報量 ≠ 性能向上」でしたが、**修正後の公平な比較**では**GPT-4 Disease-Informed が最優秀**を示しました。

### この逆転の理由
1. **実験バイアス除去**: 同一データサブセットによる真の性能測定
2. **マジックナンバー除去**: GPT-4の純粋なドメイン知識活用
3. **統計的厳密性**: K-fold CVによる信頼性の高い評価

### 実用的示唆
1. **疾患特異的情報の価値**: NSCLCの文脈情報が効果的
2. **LLMの可能性**: 適切な設計でドメイン知識を活用可能
3. **実験設計の重要性**: 公平な比較の必要性

## ファイル構成

### 実験コード
- `src/ae_exp2/experiments/main.py` - K-fold CV実験
- `src/ae_exp2/config/experiment_config.py` - 実験設定
- `src/ae_exp2/analysis/statistical_analysis.py` - 統計分析

### データと結果  
- `data.csv` - 125施設のAEデータ
- `results/data/cv_5fold_results_*.csv` - K-fold CV結果
- `results/data/cv_5fold_summary_*.csv` - CV要約統計
- `results/data/gpt4_*_response.json` - GPT-4回答ログ
- `results/plots/` - 統計分析可視化
- `results/reports/` - 包括的分析レポート

### レポート
- `results/reports/final_conclusion_report.md` - 最終結論レポート
- `results/reports/statistical_analysis_report.txt` - 統計解析レポート
- `PROGRESSIVE_INFORMATION_ANALYSIS.md` - 初期段階的実験結果

## セットアップと実行

```bash
# 依存関係のインストール
rye sync

# 環境変数の設定
cp .env.example .env
# OpenAI API keyを.envに設定

# K-fold CV実験の実行
rye run python src/ae_exp2/experiments/main.py

# 統計分析の実行
rye run python src/ae_exp2/analysis/statistical_analysis.py
```

## プロジェクト構造

```
ae_exp2/
├── src/ae_exp2/           # メインパッケージ
│   ├── experiments/       # 実験モジュール
│   │   └── main.py       # K-fold CV実験
│   ├── analysis/          # 分析モジュール
│   │   └── statistical_analysis.py
│   └── config/           # 設定モジュール
│       └── experiment_config.py
├── results/              # 実験結果
│   ├── data/            # CSV/JSON結果
│   ├── plots/           # 可視化
│   └── reports/         # 分析レポート
└── data.csv             # 元データ
```

## 今後の研究方向

1. **複数実行での安定性検証**: LLMの変動性評価
2. **他のLLMとの比較**: Claude, Gemini等
3. **ハイブリッドアプローチ**: LLM + データ探索
4. **リアルタイム事前分布更新**: 適応的ベイズ手法

## 依存関係

- Python 3.11
- PyMC (ベイズモデリング)
- ArviZ (ベイズ分析)
- OpenAI (GPT-4 API)
- NumPy, Pandas, Matplotlib, Seaborn
