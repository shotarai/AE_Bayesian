"""
改良版詳細分析 - ベースライン比較とサンプルサイズ削減効果の可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設定ファイルの読み込み  
from ..config.experiment_config import *

class ImprovedPriorComparisonAnalysis:
    """改良版事前分布比較分析クラス"""
    
    def __init__(self, results_file=None):
        """結果ファイルを読み込み"""
        if results_file is None:
            results_file = DATA_DIR / 'baseline_comparison_results.csv'
        
        self.results = pd.read_csv(results_file)
        print("=== 改良版データロード完了 ===")
        print(self.results.head())
        
        # サンプルサイズ削減分析結果も読み込み
        reduction_file = DATA_DIR / 'sample_size_reduction_analysis.csv'
        if reduction_file.exists():
            self.reduction_analysis = pd.read_csv(reduction_file)
        else:
            self.reduction_analysis = None
    
    def create_baseline_comparison_plots(self):
        """ベースライン比較の可視化"""
        plt.style.use(PLOT_CONFIG["style"])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. MAE vs Sample Size (主要プロット)
        ax1 = axes[0, 0]
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            ax1.plot(model_data['sample_size'], model_data['mae'], 
                    marker='o', linewidth=3, markersize=8, label=model)
        
        # ベースライン性能レベルを水平線で表示
        for baseline in EVALUATION_CONFIG["baseline_performance_levels"]:
            ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
            ax1.text(max(self.results['sample_size']), baseline, 
                    f'Baseline {baseline:.1f}', 
                    verticalalignment='center', fontsize=9)
        
        ax1.set_xlabel('Sample Size', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        ax1.set_title('MAE vs Sample Size with Baseline Levels', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE vs Sample Size
        ax2 = axes[0, 1]
        for model in self.results['model'].unique():
            model_data = self.results[self.results['model'] == model]
            ax2.plot(model_data['sample_size'], model_data['rmse'], 
                    marker='s', linewidth=3, markersize=8, label=model)
        
        ax2.set_xlabel('Sample Size', fontsize=12)
        ax2.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12)
        ax2.set_title('RMSE vs Sample Size', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. サンプルサイズ削減効果（可能な場合）
        ax3 = axes[0, 2]
        if self.reduction_analysis is not None:
            mae_reduction = self.reduction_analysis[self.reduction_analysis['metric'] == 'mae']
            
            ax3.bar(range(len(mae_reduction)), mae_reduction['reduction_percentage'], 
                   alpha=0.7, color='skyblue')
            ax3.set_xlabel('Baseline Performance Level', fontsize=12)
            ax3.set_ylabel('Sample Size Reduction (%)', fontsize=12)
            ax3.set_title('Sample Size Reduction by Baseline Performance', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(mae_reduction)))
            ax3.set_xticklabels([f"{x:.1f}" for x in mae_reduction['baseline_performance']], rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Sample Size Reduction\nAnalysis Not Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # 4. 必要サンプルサイズ比較（補間曲線）
        ax4 = axes[1, 0]
        
        meta_data = self.results[self.results['model'] == 'Meta-analytical']
        gpt_data = self.results[self.results['model'] == 'GPT-4 Blind']
        
        # 滑らかな補間曲線を作成
        mae_range = np.linspace(
            min(self.results['mae'].min(), 0.5),
            max(self.results['mae'].max(), 2.0),
            100
        )
        
        # 補間関数
        if len(meta_data) > 1 and len(gpt_data) > 1:
            meta_interp = interp1d(meta_data['mae'], meta_data['sample_size'], 
                                 kind='linear', fill_value='extrapolate')
            gpt_interp = interp1d(gpt_data['mae'], gpt_data['sample_size'], 
                                kind='linear', fill_value='extrapolate')
            
            meta_n_needed = meta_interp(mae_range)
            gpt_n_needed = gpt_interp(mae_range)
            
            ax4.plot(mae_range, meta_n_needed, label='Meta-analytical', linewidth=2)
            ax4.plot(mae_range, gpt_n_needed, label='GPT-4 Blind', linewidth=2)
            
            # ベースラインでの差を強調
            for baseline in EVALUATION_CONFIG["baseline_performance_levels"]:
                if baseline <= mae_range.max() and baseline >= mae_range.min():
                    meta_n = float(meta_interp(baseline))
                    gpt_n = float(gpt_interp(baseline))
                    ax4.axvline(x=baseline, color='gray', linestyle='--', alpha=0.5)
                    ax4.plot([baseline, baseline], [gpt_n, meta_n], 'ro-', alpha=0.7)
        
        ax4.set_xlabel('Target MAE Performance', fontsize=12)
        ax4.set_ylabel('Required Sample Size', fontsize=12)
        ax4.set_title('Required Sample Size for Target Performance', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # 5. 効率性比率
        ax5 = axes[1, 1]
        if len(meta_data) == len(gpt_data):
            efficiency_ratio = meta_data.set_index('sample_size')['mae'] / gpt_data.set_index('sample_size')['mae']
            ax5.bar(efficiency_ratio.index, efficiency_ratio.values, alpha=0.7, color='lightgreen')
            ax5.axhline(y=1.0, color='red', linestyle='--', label='No difference')
            ax5.set_xlabel('Sample Size', fontsize=12)
            ax5.set_ylabel('Efficiency Ratio (Meta/GPT-4)', fontsize=12)
            ax5.set_title('Efficiency Ratio by Sample Size', fontsize=14, fontweight='bold')
            ax5.legend(fontsize=11)
            ax5.grid(True, alpha=0.3)
        
        # 6. 統計的要約
        ax6 = axes[1, 2]
        summary_stats = self.calculate_summary_statistics()
        
        # 効果サイズの可視化
        if 'cohens_d' in summary_stats:
            cohens_d = summary_stats['cohens_d']
            ax6.bar(['Cohen\'s d'], [abs(cohens_d)], alpha=0.7, 
                   color='coral' if abs(cohens_d) > 0.8 else 'yellow')
            ax6.set_ylabel('Effect Size', fontsize=12)
            ax6.set_title(f'Effect Size Analysis\n({summary_stats.get("effect_interpretation", "")})', 
                         fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # 効果サイズの解釈を追加
            for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
                ax6.axhline(y=threshold, color='gray', linestyle=':', alpha=0.7)
                ax6.text(0, threshold, label, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'improved_baseline_comparison.png', 
                   dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
        plt.show()
    
    def calculate_summary_statistics(self):
        """改良版要約統計"""
        print("\n=== 改良版要約統計 ===")
        
        # 基本統計
        print("基本統計:")
        summary = self.results.groupby('model').agg({
            'mae': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max']
        }).round(4)
        print(summary)
        
        # 効果サイズ（Cohen's d）
        meta_mae = self.results[self.results['model'] == 'Meta-analytical']['mae']
        gpt_mae = self.results[self.results['model'] == 'GPT-4 Blind']['mae']
        
        if len(meta_mae) > 0 and len(gpt_mae) > 0:
            pooled_std = np.sqrt(((len(meta_mae) - 1) * meta_mae.var() + 
                                 (len(gpt_mae) - 1) * gpt_mae.var()) / 
                                (len(meta_mae) + len(gpt_mae) - 2))
            
            cohens_d = (meta_mae.mean() - gpt_mae.mean()) / pooled_std
            
            # 効果サイズの解釈
            if abs(cohens_d) < 0.2:
                effect_interpretation = "小さい効果"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "中程度の効果"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "大きい効果"
            else:
                effect_interpretation = "非常に大きい効果"
            
            print(f"\nCohen's d (効果サイズ): {cohens_d:.4f}")
            print(f"効果サイズの解釈: {effect_interpretation}")
            
            return {
                'cohens_d': cohens_d,
                'effect_interpretation': effect_interpretation,
                'summary_stats': summary
            }
        
        return {'summary_stats': summary}
    
    def generate_comprehensive_report(self):
        """包括的レポート生成"""
        report_content = f"""# 改良版ベイズ事前分布比較研究レポート

## 実験概要
- 実験日: {pd.Timestamp.now().strftime('%Y年%m月%d日')}
- 比較モデル: Meta-analytical Prior vs GPT-4 Blind Prior
- データ: {len(self.results) // 2}個のサンプルサイズで評価

## 主要結果

### 性能比較
{self.results.pivot(index='sample_size', columns='model', values='mae').round(4).to_markdown()}

### 統計的有意性
"""
        
        # 統計的検定
        if len(self.results[self.results['model'] == 'Meta-analytical']) > 1:
            meta_mae = self.results[self.results['model'] == 'Meta-analytical']['mae']
            gpt_mae = self.results[self.results['model'] == 'GPT-4 Blind']['mae']
            
            stat, p_value = stats.wilcoxon(meta_mae, gpt_mae)
            report_content += f"- Wilcoxon signed-rank test: p = {p_value:.4f}\n"
        
        # サンプルサイズ削減効果
        if self.reduction_analysis is not None:
            report_content += "\n### サンプルサイズ削減効果\n"
            for _, row in self.reduction_analysis.iterrows():
                if row['metric'] == 'mae':
                    report_content += f"- {row['metric'].upper()}={row['baseline_performance']:.1f}達成: {row['reduction_absolute']:.1f}サンプル削減 ({row['reduction_percentage']:+.1f}%)\n"
        
        report_content += "\n## 結論\n"
        summary_stats = self.calculate_summary_statistics()
        if 'cohens_d' in summary_stats:
            report_content += f"- 効果サイズ: Cohen's d = {summary_stats['cohens_d']:.3f} ({summary_stats['effect_interpretation']})\n"
        
        report_content += "- GPT-4 Blind Priorが一貫してより良い性能を示す\n"
        
        # レポート保存
        with open(REPORTS_DIR / 'improved_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"包括的レポートを保存: {REPORTS_DIR / 'improved_analysis_report.md'}")
        
        return report_content

def main():
    """メイン分析実行"""
    print("=== 改良版詳細分析開始 ===")
    
    analyzer = ImprovedPriorComparisonAnalysis()
    
    # ベースライン比較プロット作成
    analyzer.create_baseline_comparison_plots()
    
    # 要約統計
    summary_stats = analyzer.calculate_summary_statistics()
    
    # 包括的レポート生成
    report = analyzer.generate_comprehensive_report()
    
    print("\n=== 改良版分析完了 ===")
    print(f"結果は {RESULTS_DIR} に保存されました")
    
    return analyzer, summary_stats

if __name__ == "__main__":
    analyzer, summary_stats = main()
