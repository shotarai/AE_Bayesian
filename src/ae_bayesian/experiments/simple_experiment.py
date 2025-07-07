"""
Simplified Bayesian Prior Comparison Experiment
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import json
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class SimpleBayesianExperiment:
    """Simplified adverse event (AE) reporting rate Bayesian experiment class"""
    
    def __init__(self, data_path='data.csv'):
        """Initialize the experiment"""
        self.data = pd.read_csv(data_path)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.process_data()
        
    def process_data(self):
        """Perform data preprocessing"""
        # Calculate AE reporting rate for each site
        self.site_stats = self.data.groupby('site_number').agg({
            'patnum': 'nunique',
            'ae_count_cumulative': 'sum'
        }).reset_index()
        
        # Calculate AE reporting rate (λ)
        self.site_stats['ae_rate'] = (
            self.site_stats['ae_count_cumulative'] / 
            self.site_stats['patnum']
        )
        
        print(f"Total sites: {len(self.site_stats)}")
        print(f"AE rate statistics:")
        print(self.site_stats['ae_rate'].describe())
    
    def get_gpt4_prior(self):
        """Obtain prior distribution parameters from GPT-4"""
        print("\n=== Querying GPT-4 for Prior Distribution ===")
        
        data_summary = f"""
        Data Summary:
        - Number of sites: {len(self.site_stats)}
        - Mean AE rate: {self.site_stats['ae_rate'].mean():.3f}
        - Std AE rate: {self.site_stats['ae_rate'].std():.3f}
        """
        
        prompt = f"""
        You are a biostatistics expert. I need prior parameters for a hierarchical Bayesian model:
        - Each site's AE rate λ_i ~ Gamma(α, β)
        - Need priors for α and β hyperparameters
        
        {data_summary}
        
        Provide specific numerical values in JSON format:
        {{
            "alpha_prior": {{"distribution": "gamma", "parameters": [alpha_shape, alpha_rate]}},
            "beta_prior": {{"distribution": "gamma", "parameters": [beta_shape, beta_rate]}},
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a biostatistics expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            print("GPT-4 Response:")
            print(response_text)
            
            # Extract JSON response
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found")
            
            gpt_priors = json.loads(json_text)
            return gpt_priors
            
        except Exception as e:
            print(f"Error with GPT-4: {e}")
            return {
                "alpha_prior": {"distribution": "gamma", "parameters": [2.0, 1.0]},
                "beta_prior": {"distribution": "gamma", "parameters": [2.0, 10.0]},
                "reasoning": "Fallback priors"
            }
    
    def run_model_a(self, n_samples=20):
        """Model A: Meta-analytical prior"""
        print(f"\n=== Model A: Meta-analytical Prior (n={n_samples}) ===")
        
        sample_data = self.site_stats.sample(n=min(n_samples, len(self.site_stats)), 
                                           random_state=42)
        
        observed_ae = sample_data['ae_count_cumulative'].values
        observed_patients = sample_data['patnum'].values
        n_sites = len(sample_data)
        
        with pm.Model() as model_a:
            # Non-informative priors
            alpha_study = pm.Exponential('alpha_study', lam=0.1)
            beta_study = pm.Exponential('beta_study', lam=0.1)
            
            # AE reporting rate for each site
            lambda_sites = pm.Gamma('lambda_sites', 
                                  alpha=alpha_study, 
                                  beta=beta_study, 
                                  shape=n_sites)
            
            # Likelihood
            ae_counts = pm.Poisson('ae_counts', 
                                 mu=lambda_sites * observed_patients,
                                 observed=observed_ae)
            
            # Sampling
            trace_a = pm.sample(500, tune=500, return_inferencedata=True, 
                              random_seed=42, progressbar=False)
        
        return model_a, trace_a, sample_data
    
    def run_model_b(self, n_samples=20, gpt_priors=None):
        """Model B: GPT-4 prior"""
        print(f"\n=== Model B: GPT-4 Prior (n={n_samples}) ===")
        
        if gpt_priors is None:
            gpt_priors = self.get_gpt4_prior()
        
        sample_data = self.site_stats.sample(n=min(n_samples, len(self.site_stats)), 
                                           random_state=42)
        
        observed_ae = sample_data['ae_count_cumulative'].values
        observed_patients = sample_data['patnum'].values
        n_sites = len(sample_data)
        
        with pm.Model() as model_b:
            # GPT-4 recommended prior distribution
            alpha_prior = gpt_priors['alpha_prior']
            beta_prior = gpt_priors['beta_prior']
            
            alpha_study = pm.Gamma('alpha_study', 
                                 alpha=alpha_prior['parameters'][0],
                                 beta=alpha_prior['parameters'][1])
            beta_study = pm.Gamma('beta_study',
                                alpha=beta_prior['parameters'][0],
                                beta=beta_prior['parameters'][1])
            
            # AE reporting rate for each site
            lambda_sites = pm.Gamma('lambda_sites',
                                  alpha=alpha_study,
                                  beta=beta_study,
                                  shape=n_sites)
            
            # Likelihood
            ae_counts = pm.Poisson('ae_counts',
                                 mu=lambda_sites * observed_patients,
                                 observed=observed_ae)
            
            # Sampling
            trace_b = pm.sample(500, tune=500, return_inferencedata=True,
                              random_seed=42, progressbar=False)
        
        return model_b, trace_b, sample_data, gpt_priors
    
    def evaluate_model(self, model, trace, sample_data):
        """Model evaluation"""
        with model:
            ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=False)
        
        observed = sample_data['ae_count_cumulative'].values
        predicted = ppc.posterior_predictive['ae_counts'].values
        pred_mean = np.mean(predicted, axis=(0, 1))
        
        # Calculate performance metrics
        mae = np.mean(np.abs(observed - pred_mean))
        rmse = np.sqrt(np.mean((observed - pred_mean)**2))
        
        # Log predictive density
        lpd = 0
        for i, obs in enumerate(observed):
            pred_lambda = sample_data.iloc[i]['patnum'] * np.mean(trace.posterior['lambda_sites'].values[:, :, i])
            if obs < 100:
                lpd += -pred_lambda + obs * np.log(pred_lambda + 1e-10) - gammaln(obs + 1)
            else:
                lpd += -1e10
        
        return {'mae': mae, 'rmse': rmse, 'lpd': lpd}
    
    def compare_models(self, sample_sizes=[10, 20, 30]):
        """Model comparison"""
        print("\n=== Model Comparison ===")
        
        # Obtain GPT-4 prior in advance
        gpt_priors = self.get_gpt4_prior()
        
        results = []
        
        for n_samples in sample_sizes:
            print(f"\n--- Sample size: {n_samples} ---")
            
            try:
                # Model A
                model_a, trace_a, sample_data_a = self.run_model_a(n_samples)
                perf_a = self.evaluate_model(model_a, trace_a, sample_data_a)
                
                # Model B
                model_b, trace_b, sample_data_b, _ = self.run_model_b(n_samples, gpt_priors)
                perf_b = self.evaluate_model(model_b, trace_b, sample_data_b)
                
                results.append({
                    'sample_size': n_samples,
                    'model': 'Meta-analytical',
                    'mae': perf_a['mae'],
                    'rmse': perf_a['rmse'],
                    'lpd': perf_a['lpd']
                })
                
                results.append({
                    'sample_size': n_samples,
                    'model': 'GPT-4',
                    'mae': perf_b['mae'],
                    'rmse': perf_b['rmse'],
                    'lpd': perf_b['lpd']
                })
                
                print(f"Model A - MAE: {perf_a['mae']:.3f}, RMSE: {perf_a['rmse']:.3f}, LPD: {perf_a['lpd']:.3f}")
                print(f"Model B - MAE: {perf_b['mae']:.3f}, RMSE: {perf_b['rmse']:.3f}, LPD: {perf_b['lpd']:.3f}")
                
            except Exception as e:
                print(f"Error with sample size {n_samples}: {e}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create plots
        if not results_df.empty:
            self.plot_results(results_df)
        
        return results_df
    
    def plot_results(self, results_df):
        """Plot results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['mae', 'rmse', 'lpd']
        titles = ['Mean Absolute Error', 'Root Mean Square Error', 'Log Predictive Density']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                ax.plot(model_data['sample_size'], model_data[metric], 
                       marker='o', label=model, linewidth=2)
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== Results Summary ===")
        print(results_df.to_string(index=False))
        
        return results_df

def main():
    """Main execution function"""
    print("=== Simple Bayesian Prior Comparison ===")
    
    experiment = SimpleBayesianExperiment()
    
    # Execute model comparison
    results_df = experiment.compare_models([10, 20, 30])
    
    # Save results
    if not results_df.empty:
        results_df.to_csv('simple_comparison_results.csv', index=False)
        print("\nResults saved to 'simple_comparison_results.csv'")
    
    print("\n=== Experiment Complete ===")

if __name__ == "__main__":
    main()
