"""
改良版実験: マジックナンバー排除とベースライン比較分析
"""

import os
import sys
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path
from scipy.special import gammaln
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# 設定ファイルの読み込み
from ..config.experiment_config import *

load_dotenv()

class ImprovedBayesianExperiment:
    """改良版ベイズ実験クラス - マジックナンバー排除とベースライン比較"""
    
    def __init__(self, data_path='data.csv'):
        """初期化"""
        self.data = pd.read_csv(data_path)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.gpt_priors = None
        self.process_data()
        
        print(f"結果保存先: {RESULTS_DIR}")
        
    def process_data(self):
        """データの前処理"""
        self.site_stats = self.data.groupby('site_number').agg({
            'patnum': 'nunique',
            'ae_count_cumulative': 'sum'
        }).reset_index()
        
        self.site_stats['ae_rate'] = (
            self.site_stats['ae_count_cumulative'] / 
            self.site_stats['patnum']
        )
        
        print(f"Total sites: {len(self.site_stats)}")
        print(f"AE rate statistics:")
        print(self.site_stats['ae_rate'].describe())
    
    def get_gpt4_prior_completely_blind(self):
        """GPT-4から完全にブラインドで事前分布を取得（データ情報なし）"""
        print("\n=== GPT-4への完全ブラインド問い合わせ ===")
        
        prompt = """
        You are a biostatistics expert specializing in clinical trials and Bayesian analysis.
        
        I need you to recommend prior distributions for a hierarchical Bayesian model of adverse event (AE) reporting rates in clinical trials.
        
        Model structure:
        - Each site's AE rate λ_i ~ Gamma(α, β)  
        - Need priors for hyperparameters α and β
        
        Important: I am NOT providing you with any specific data. Please recommend priors based purely on:
        1. General knowledge of clinical trial adverse event rates
        2. Typical ranges you would expect in pharmaceutical studies
        3. Bayesian modeling best practices
        4. Your understanding of regulatory requirements
        
        Please avoid any "magic numbers" or arbitrary values. Base your recommendations on domain knowledge.
        
        Provide your response in this exact JSON format:
        {
            "alpha_prior": {"distribution": "gamma", "parameters": [shape, rate]},
            "beta_prior": {"distribution": "gamma", "parameters": [shape, rate]},
            "reasoning": "detailed explanation of your choices based on clinical domain knowledge",
            "expected_ae_rate_range": "expected range of AE rates you're assuming",
            "confidence_level": "how confident you are in these priors (low/medium/high)"
        }
        
        For gamma distribution: Gamma(shape, rate) where mean = shape/rate
        """
        
        for attempt in range(GPT4_CONFIG["max_retries"]):
            try:
                print(f"GPT-4問い合わせ試行 {attempt + 1}/{GPT4_CONFIG['max_retries']}")
                
                response = self.client.chat.completions.create(
                    model=GPT4_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "You are a biostatistics expert with deep knowledge of clinical trials and regulatory requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=GPT4_CONFIG["max_tokens"],
                    temperature=GPT4_CONFIG["temperature"]
                )
                
                response_text = response.choices[0].message.content
                print("GPT-4完全ブラインド回答:")
                print(response_text)
                
                # JSONレスポンスを抽出
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    raise ValueError("JSONが見つかりません")
                
                gpt_priors = json.loads(json_text)
                
                # 回答を保存
                with open(DATA_DIR / "gpt4_blind_response.json", 'w', encoding='utf-8') as f:
                    json.dump({
                        "prompt": prompt,
                        "response": response_text,
                        "extracted_priors": gpt_priors,
                        "attempt": attempt + 1
                    }, f, indent=2, ensure_ascii=False)
                
                return gpt_priors
                
            except Exception as e:
                print(f"GPT-4問い合わせエラー (試行 {attempt + 1}): {e}")
                if attempt == GPT4_CONFIG["max_retries"] - 1:
                    print("GPT-4が利用できません。完全非情報的事前分布を使用します。")
                    return MODEL_CONFIG["gpt4_fallback"]
        
        return MODEL_CONFIG["gpt4_fallback"]
    
    def run_model_meta_analytical(self, n_samples=20):
        """モデルA: Meta-analytical prior (Barmaz & Ménard 2021)"""
        print(f"\n=== Meta-analytical Prior (n={n_samples}) ===")
        
        sample_data = self.site_stats.sample(
            n=min(n_samples, len(self.site_stats)), 
            random_state=EXPERIMENT_CONFIG["random_seed"]
        )
        
        observed_ae = sample_data['ae_count_cumulative'].values
        observed_patients = sample_data['patnum'].values
        n_sites = len(sample_data)
        
        with pm.Model() as model_meta:
            # Barmaz & Ménard (2021) 非情報的事前分布
            config = MODEL_CONFIG["meta_analytical"]
            alpha_study = pm.Exponential('alpha_study', 
                                       lam=config["alpha_prior"]["parameters"][0])
            beta_study = pm.Exponential('beta_study', 
                                      lam=config["beta_prior"]["parameters"][0])
            
            # 各施設のAE報告率
            lambda_sites = pm.Gamma('lambda_sites', 
                                  alpha=alpha_study, 
                                  beta=beta_study, 
                                  shape=n_sites)
            
            # 尤度
            ae_counts = pm.Poisson('ae_counts', 
                                 mu=lambda_sites * observed_patients,
                                 observed=observed_ae)
            
            # サンプリング
            trace_meta = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"], 
                tune=EXPERIMENT_CONFIG["mcmc_tune"], 
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"], 
                progressbar=False
            )
        
        return model_meta, trace_meta, sample_data
    
    def run_model_gpt4(self, n_samples=20):
        """モデルB: GPT-4完全ブラインド事前分布"""
        print(f"\n=== GPT-4 Blind Prior (n={n_samples}) ===")
        
        if self.gpt_priors is None:
            self.gpt_priors = self.get_gpt4_prior_completely_blind()
        
        sample_data = self.site_stats.sample(
            n=min(n_samples, len(self.site_stats)), 
            random_state=EXPERIMENT_CONFIG["random_seed"]
        )
        
        observed_ae = sample_data['ae_count_cumulative'].values
        observed_patients = sample_data['patnum'].values
        n_sites = len(sample_data)
        
        with pm.Model() as model_gpt:
            # GPT-4完全ブラインド推奨事前分布
            alpha_prior = self.gpt_priors['alpha_prior']
            beta_prior = self.gpt_priors['beta_prior']
            
            print(f"GPT-4推奨事前分布: α ~ Gamma{tuple(alpha_prior['parameters'])}, β ~ Gamma{tuple(beta_prior['parameters'])}")
            
            alpha_study = pm.Gamma('alpha_study', 
                                 alpha=alpha_prior['parameters'][0],
                                 beta=alpha_prior['parameters'][1])
            beta_study = pm.Gamma('beta_study',
                                alpha=beta_prior['parameters'][0],
                                beta=beta_prior['parameters'][1])
            
            # 各施設のAE報告率
            lambda_sites = pm.Gamma('lambda_sites',
                                  alpha=alpha_study,
                                  beta=beta_study,
                                  shape=n_sites)
            
            # 尤度
            ae_counts = pm.Poisson('ae_counts',
                                 mu=lambda_sites * observed_patients,
                                 observed=observed_ae)
            
            # サンプリング
            trace_gpt = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"], 
                tune=EXPERIMENT_CONFIG["mcmc_tune"], 
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"], 
                progressbar=False
            )
        
        return model_gpt, trace_gpt, sample_data
    
    def calculate_predictive_performance(self, trace, sample_data):
        """予測性能の計算"""
        # 実際のAE率
        true_rates = sample_data['ae_rate'].values
        
        # 事後予測分布から予測値を取得
        posterior_lambda = trace.posterior['lambda_sites'].values
        # shape: (chains, draws, sites)
        predicted_rates = posterior_lambda.mean(axis=(0, 1))
        
        # MAE
        mae = np.mean(np.abs(predicted_rates - true_rates))
        
        # RMSE  
        rmse = np.sqrt(np.mean((predicted_rates - true_rates) ** 2))
        
        # Log Predictive Density (簡略版)
        log_pred_density = np.mean([
            np.log(np.mean(np.exp(-0.5 * ((pred - true) ** 2))))
            for pred, true in zip(predicted_rates, true_rates)
        ])
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'lpd': log_pred_density,
            'predicted_rates': predicted_rates,
            'true_rates': true_rates
        }
    
    def run_baseline_comparison_experiment(self):
        """ベースライン比較実験"""
        print("\n" + "="*60)
        print("ベースライン比較実験開始")
        print("="*60)
        
        results = []
        sample_sizes = EXPERIMENT_CONFIG["sample_sizes"]
        
        for sample_size in sample_sizes:
            print(f"\n--- サンプルサイズ {sample_size} ---")
            
            # Meta-analytical model
            _, trace_meta, sample_data = self.run_model_meta_analytical(sample_size)
            perf_meta = self.calculate_predictive_performance(trace_meta, sample_data)
            
            results.append({
                'sample_size': sample_size,
                'model': 'Meta-analytical',
                'mae': perf_meta['mae'],
                'rmse': perf_meta['rmse'],
                'lpd': perf_meta['lpd']
            })
            
            # GPT-4 model
            _, trace_gpt, _ = self.run_model_gpt4(sample_size)
            perf_gpt = self.calculate_predictive_performance(trace_gpt, sample_data)
            
            results.append({
                'sample_size': sample_size,
                'model': 'GPT-4 Blind',
                'mae': perf_gpt['mae'],
                'rmse': perf_gpt['rmse'],
                'lpd': perf_gpt['lpd']
            })
            
            print(f"Meta-analytical - MAE: {perf_meta['mae']:.4f}, RMSE: {perf_meta['rmse']:.4f}")
            print(f"GPT-4 Blind     - MAE: {perf_gpt['mae']:.4f}, RMSE: {perf_gpt['rmse']:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(DATA_DIR / 'baseline_comparison_results.csv', index=False)
        
        return results_df
    
    def analyze_sample_size_reduction(self, results_df):
        """ベースライン性能での必要サンプルサイズ削減効果を分析"""
        print("\n=== サンプルサイズ削減効果分析 ===")
        
        # Meta-analyticalとGPT-4のデータを分離
        meta_data = results_df[results_df['model'] == 'Meta-analytical'].copy()
        gpt_data = results_df[results_df['model'] == 'GPT-4 Blind'].copy()
        
        reduction_analysis = []
        
        for metric in ['mae', 'rmse']:
            print(f"\n--- {metric.upper()}による分析 ---")
            
            # 補間関数を作成
            meta_interp = interp1d(meta_data[metric], meta_data['sample_size'], 
                                 kind='linear', fill_value='extrapolate')
            gpt_interp = interp1d(gpt_data[metric], gpt_data['sample_size'], 
                                kind='linear', fill_value='extrapolate')
            
            # 各ベースライン性能レベルでの必要サンプルサイズを比較
            for baseline_performance in EVALUATION_CONFIG["baseline_performance_levels"]:
                try:
                    # その性能を達成するのに必要なサンプルサイズ
                    meta_required_n = float(meta_interp(baseline_performance))
                    gpt_required_n = float(gpt_interp(baseline_performance))
                    
                    # サンプルサイズ削減効果
                    if meta_required_n > 0:
                        reduction_pct = (meta_required_n - gpt_required_n) / meta_required_n * 100
                        reduction_abs = meta_required_n - gpt_required_n
                        
                        reduction_analysis.append({
                            'metric': metric,
                            'baseline_performance': baseline_performance,
                            'meta_required_n': meta_required_n,
                            'gpt_required_n': gpt_required_n,
                            'reduction_absolute': reduction_abs,
                            'reduction_percentage': reduction_pct
                        })
                        
                        print(f"  {metric.upper()}={baseline_performance:.2f}の性能達成に必要なサンプルサイズ:")
                        print(f"    Meta-analytical: {meta_required_n:.1f}")
                        print(f"    GPT-4 Blind:     {gpt_required_n:.1f}")
                        print(f"    削減効果:        {reduction_abs:.1f} ({reduction_pct:+.1f}%)")
                        
                except Exception as e:
                    print(f"  {metric.upper()}={baseline_performance:.2f}: 計算できません ({e})")
        
        reduction_df = pd.DataFrame(reduction_analysis)
        reduction_df.to_csv(DATA_DIR / 'sample_size_reduction_analysis.csv', index=False)
        
        return reduction_df

def main():
    """メイン実験実行"""
    print("改良版ベイズ事前分布比較実験")
    print(f"結果保存先: {RESULTS_DIR}")
    
    # 実験実行
    experiment = ImprovedBayesianExperiment()
    
    # ベースライン比較実験
    results_df = experiment.run_baseline_comparison_experiment()
    
    # サンプルサイズ削減効果分析
    reduction_df = experiment.analyze_sample_size_reduction(results_df)
    
    print(f"\n実験完了！結果は {RESULTS_DIR} に保存されました。")
    
    return results_df, reduction_df

if __name__ == "__main__":
    results_df, reduction_df = main()
