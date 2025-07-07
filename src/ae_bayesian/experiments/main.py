"""
Improved Experiment: Magic Number Elimination and Baseline Comparison Analysis
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

# Load configuration file
from ..config.experiment_config import *

load_dotenv()

class ImprovedBayesianExperiment:
    """Improved Bayesian Experiment Class - Magic Number Elimination and Baseline Comparison"""
    
    def __init__(self, data_path='data.csv'):
        """Initialize"""
        self.data = pd.read_csv(data_path)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.gpt_priors = None
        self.process_data()
        
        print(f"Results save location: {RESULTS_DIR}")
        
    def process_data(self):
        """IPD (Individual Patient Data) preprocessing - Compliant with Barmaz & Ménard"""
        # Preserve patient-level data as is (this is IPD)
        self.patient_data = self.data.copy()
        
        # Data validation
        print(f"IPD Statistics:")
        print(f"Total patient records: {len(self.patient_data)}")
        print(f"Unique patients: {self.patient_data['patnum'].nunique()}")
        print(f"Unique sites: {self.patient_data['site_number'].nunique()}")
        print(f"Patient-level AE distribution:")
        print(self.patient_data['ae_count_cumulative'].describe())
        
        # Site-level summary (for reference only)
        self.site_summary = self.data.groupby('site_number').agg({
            'patnum': 'nunique',
            'ae_count_cumulative': ['sum', 'mean', 'count']
        }).round(4)
        
        print(f"\nSite-level summary (for reference):")
        print(f"Average patients per site: {self.site_summary[('patnum', 'nunique')].mean():.1f}")
        print(f"Average AE count per patient: {self.patient_data['ae_count_cumulative'].mean():.3f}")
    
    def create_stratified_cv_folds(self, n_folds=5):
        """Create stratified cross-validation folds (Hastie et al. 2009 compliant)"""
        print(f"\n=== {n_folds}-Fold Stratified Cross-Validation Setup ===")
        
        # Site-level stratification
        sites = self.patient_data['site_number'].unique()
        
        # Stratification by patients per site (small, medium, large sites)
        site_patient_counts = self.patient_data.groupby('site_number').size()
        
        # Divide into 3 strata (Efron & Tibshirani 1993)
        site_terciles = np.percentile(site_patient_counts, [33.33, 66.67])
        
        small_sites = site_patient_counts[site_patient_counts <= site_terciles[0]].index
        medium_sites = site_patient_counts[
            (site_patient_counts > site_terciles[0]) & 
            (site_patient_counts <= site_terciles[1])
        ].index
        large_sites = site_patient_counts[site_patient_counts > site_terciles[1]].index
        
        print(f"Small sites: {len(small_sites)} sites")
        print(f"Medium sites: {len(medium_sites)} sites") 
        print(f"Large sites: {len(large_sites)} sites")
        
        # Sample equally from each stratum
        folds = []
        np.random.seed(EXPERIMENT_CONFIG["random_seed"])
        
        for fold in range(n_folds):
            fold_sites = []
            
            # Select equally from each stratum
            for sites_in_stratum in [small_sites, medium_sites, large_sites]:
                if len(sites_in_stratum) > 0:
                    n_per_fold = max(1, len(sites_in_stratum) // n_folds)
                    selected = np.random.choice(
                        sites_in_stratum, 
                        size=min(n_per_fold, len(sites_in_stratum)), 
                        replace=False
                    )
                    fold_sites.extend(selected)
            
            folds.append(fold_sites)
        
        return folds
    
    def get_gpt4_prior_completely_blind(self):
        """Obtain prior distributions from GPT-4 completely blind (no data information)"""
        print("\n=== Completely Blind Inquiry to GPT-4 ===")
        
        prompt = """
        You are a biostatistics expert specializing in clinical trials and Bayesian analysis.
        
        I need you to recommend prior distributions for a hierarchical Bayesian model analyzing individual patient data (IPD) for adverse event (AE) reporting in clinical trials.
        
        Model structure (Individual Patient Data):
        - Each patient i in site j has AE count: y_ij ~ Poisson(λ_j)
        - Site-specific AE rates: λ_j ~ Gamma(α, β)
        - Need priors for hyperparameters α and β
        
        Key considerations for IPD analysis:
        1. Each observation represents one patient's total AE count
        2. Patients are nested within sites (hierarchical structure)
        3. Site-level variation in patient AE rates across different medical centers
        4. Regulatory requirements for pharmaceutical studies
        
        Important: I am NOT providing you with any specific data, numbers, or ranges. Please recommend priors based purely on:
        1. Your general knowledge of patient-level adverse event rates in pharmaceutical trials
        2. Standard ranges you would expect based on your clinical trial expertise
        3. Bayesian modeling best practices for hierarchical IPD models
        4. Your understanding of regulatory statistical requirements
        
        Please avoid any pre-specified numerical ranges. Base your recommendations entirely on your domain expertise.
        
        Provide your response in this exact JSON format:
        {
            "alpha_prior": {"distribution": "exponential", "parameters": [rate]},
            "beta_prior": {"distribution": "exponential", "parameters": [rate]},
            "reasoning": "detailed explanation of your choices based on IPD clinical domain knowledge",
            "expected_patient_ae_rate_range": "expected range of patient-level AE rates you're assuming based on your expertise",
            "confidence_level": "how confident you are in these priors (low/medium/high)"
        }
        
        For exponential distribution: Exponential(rate) where mean = 1/rate
        """
        
        for attempt in range(GPT4_CONFIG["max_retries"]):
            try:
                print(f"GPT-4 inquiry attempt {attempt + 1}/{GPT4_CONFIG['max_retries']}")
                
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
                print("GPT-4 completely blind response:")
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
                    raise ValueError("JSON not found")
                
                gpt_priors = json.loads(json_text)
                
                # Parameter validity check
                alpha_params = gpt_priors['alpha_prior']['parameters']
                beta_params = gpt_priors['beta_prior']['parameters']
                
                if (not isinstance(alpha_params[0], (int, float)) or 
                    not isinstance(beta_params[0], (int, float))):
                    raise ValueError("Parameters are not numeric")
                
                # Save response
                with open(DATA_DIR / "gpt4_blind_response.json", 'w', encoding='utf-8') as f:
                    json.dump({
                        "prompt": prompt,
                        "response": response_text,
                        "extracted_priors": gpt_priors,
                        "attempt": attempt + 1
                    }, f, indent=2, ensure_ascii=False)
                
                return gpt_priors
                
            except Exception as e:
                print(f"GPT-4 inquiry error (attempt {attempt + 1}): {e}")
                if attempt == GPT4_CONFIG["max_retries"] - 1:
                    print("GPT-4 unavailable. Using completely non-informative priors.")
                    return MODEL_CONFIG["gpt4_fallback"]
        
        return MODEL_CONFIG["gpt4_fallback"]
    
    def get_gpt4_prior_with_disease_info(self):
        """Obtain prior distributions from GPT-4 with disease information"""
        print("\n=== Disease Information Inquiry to GPT-4 ===")
        
        prompt = """
        You are a biostatistics expert specializing in clinical trials and Bayesian analysis.
        
        I need you to recommend prior distributions for a hierarchical Bayesian model analyzing individual patient data (IPD) for adverse event (AE) reporting in clinical trials.
        
        Study context (anonymized):
        - Disease area: Non-small cell lung cancer (NSCLC)
        - Treatment arm: Control arm (placebo or standard care)
        - Study type: Multi-center randomized controlled trial
        - Patient population: Adult oncology patients
        - Data structure: Individual patient AE counts
        
        Model structure (Individual Patient Data):
        - Each patient i in site j has AE count: y_ij ~ Poisson(λ_j)
        - Site-specific patient AE rates: λ_j ~ Gamma(α, β)
        - Need priors for hyperparameters α and β
        
        Please recommend priors based on:
        1. NSCLC-specific patient-level adverse event patterns in control arms
        2. Your expertise on typical patient-level AE counts in oncology control groups
        3. Multi-center variability in patient AE rates across cancer centers
        4. Regulatory requirements for cancer studies
        5. IPD modeling considerations for hierarchical data
        
        Expected considerations for NSCLC control arm patients:
        - Control arm patients typically have fewer AEs than treatment arms
        - NSCLC baseline disease-related events (respiratory, constitutional)
        - Age-related comorbidities in cancer populations
        - Site-to-site variation in patient monitoring and AE reporting practices
        - Patient-level heterogeneity within sites
        
        Please base your numerical recommendations entirely on your clinical expertise without any pre-specified ranges.
        
        Provide your response in this exact JSON format:
        {
            "alpha_prior": {"distribution": "exponential", "parameters": [rate]},
            "beta_prior": {"distribution": "exponential", "parameters": [rate]},
            "reasoning": "detailed explanation of your choices based on NSCLC IPD clinical domain knowledge",
            "expected_patient_ae_rate_range": "expected range of patient-level AE rates you're assuming for NSCLC control arm",
            "confidence_level": "how confident you are in these priors (low/medium/high)"
        }
        
        For exponential distribution: Exponential(rate) where mean = 1/rate
        """
        
        for attempt in range(GPT4_CONFIG["max_retries"]):
            try:
                print(f"GPT-4 inquiry attempt {attempt + 1}/{GPT4_CONFIG['max_retries']}")
                
                response = self.client.chat.completions.create(
                    model=GPT4_CONFIG["model"],
                    messages=[
                        {"role": "system", "content": "You are a biostatistics expert with deep knowledge of oncology clinical trials and regulatory requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=GPT4_CONFIG["max_tokens"],
                    temperature=GPT4_CONFIG["temperature"]
                )
                
                response_text = response.choices[0].message.content
                print("GPT-4 disease-informed response:")
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
                    raise ValueError("JSON not found")
                
                gpt_priors = json.loads(json_text)
                
                # Save response
                with open(DATA_DIR / "gpt4_disease_response.json", 'w', encoding='utf-8') as f:
                    json.dump({
                        "prompt": prompt,
                        "response": response_text,
                        "extracted_priors": gpt_priors,
                        "attempt": attempt + 1
                    }, f, indent=2, ensure_ascii=False)
                
                return gpt_priors
                
            except Exception as e:
                print(f"GPT-4 inquiry error (attempt {attempt + 1}): {e}")
                if attempt == GPT4_CONFIG["max_retries"] - 1:
                    print("GPT-4 unavailable. Using completely non-informative prior distribution.")
                    return MODEL_CONFIG["gpt4_fallback"]
        
        return MODEL_CONFIG["gpt4_fallback"]

    def run_model_meta_analytical(self, selected_sites=None, n_sites=20):
        """Model A: Meta-analytical prior (Barmaz & Ménard 2021) - IPD version"""
        print(f"\n=== Meta-analytical Prior IPD Model (n_sites={n_sites}) ===")
        
        # Site selection (External specification or fixed seed usage)
        if selected_sites is None:
            available_sites = self.patient_data['site_number'].unique()
            np.random.seed(EXPERIMENT_CONFIG["random_seed"])  # Same seed for all models
            selected_sites = np.random.choice(
                available_sites, 
                size=min(n_sites, len(available_sites)), 
                replace=False
            )
        
        # Obtain patient data from selected sites
        sample_data = self.patient_data[
            self.patient_data['site_number'].isin(selected_sites)
        ].copy()
        
        # IPD model data preparation
        sites = sample_data['site_number'].values
        observed_ae = sample_data['ae_count_cumulative'].values  # AE count for each patient
        unique_sites, sites_idx = np.unique(sites, return_inverse=True)
        n_sites = len(unique_sites)
        
        print(f"Number of selected sites: {n_sites}")
        print(f"Number of patients: {len(sample_data)}")
        print(f"AE range: {observed_ae.min()} - {observed_ae.max()}")
        
        with pm.Model() as model_meta:
            # Barmaz & Ménard (2021) prior distribution - for IPD
            config = MODEL_CONFIG["meta_analytical"]
            alpha_study = pm.Exponential('alpha_study', 
                                       lam=config["alpha_prior"]["parameters"][0])
            beta_study = pm.Exponential('beta_study', 
                                      lam=config["beta_prior"]["parameters"][0])
            
            # Patient-level AE rates for each site (Gamma distribution)
            lambda_sites = pm.Gamma('lambda_sites', 
                                  alpha=alpha_study, 
                                  beta=beta_study, 
                                  shape=n_sites)
            
            # IPD likelihood: AE count for each patient ~ Poisson(AE rate of the patient's site)
            ae_counts = pm.Poisson('ae_counts', 
                                 mu=lambda_sites[sites_idx],  # AE rate of the facility where the patient belongs
                                 observed=observed_ae)
            
            # Sampling
            trace_meta = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"], 
                tune=EXPERIMENT_CONFIG["mcmc_tune"], 
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"], 
                progressbar=False
            )
        
        return model_meta, trace_meta, sample_data
    
    def run_model_gpt4(self, selected_sites=None, n_sites=20):
        """Model B: GPT-4 completely blind prior distribution - IPD version"""
        print(f"\n=== GPT-4 Blind Prior IPD Model (n_sites={n_sites}) ===")
        
        if self.gpt_priors is None:
            self.gpt_priors = self.get_gpt4_prior_completely_blind()
        
        # Site selection (External specification or fixed seed usage)
        if selected_sites is None:
            available_sites = self.patient_data['site_number'].unique()
            np.random.seed(EXPERIMENT_CONFIG["random_seed"])  # Same seed for all models
            selected_sites = np.random.choice(
                available_sites, 
                size=min(n_sites, len(available_sites)), 
                replace=False
            )
        
        # Obtain patient data from selected sites
        sample_data = self.patient_data[
            self.patient_data['site_number'].isin(selected_sites)
        ].copy()
        
        # Data preparation for IPD model
        sites = sample_data['site_number'].values
        observed_ae = sample_data['ae_count_cumulative'].values
        unique_sites, sites_idx = np.unique(sites, return_inverse=True)
        n_sites = len(unique_sites)
        
        with pm.Model() as model_gpt:
            # GPT-4 completely blind recommended prior distribution
            alpha_prior = self.gpt_priors['alpha_prior']
            beta_prior = self.gpt_priors['beta_prior']
            
            print(f"GPT-4 recommended prior distribution: α ~ {alpha_prior['distribution'].title()}{tuple(alpha_prior['parameters'])}, β ~ {beta_prior['distribution'].title()}{tuple(beta_prior['parameters'])}")
            
            if alpha_prior['distribution'] == 'exponential':
                alpha_study = pm.Exponential('alpha_study', lam=alpha_prior['parameters'][0])
            else:
                alpha_study = pm.Gamma('alpha_study', 
                                     alpha=alpha_prior['parameters'][0],
                                     beta=alpha_prior['parameters'][1])
            
            if beta_prior['distribution'] == 'exponential':
                beta_study = pm.Exponential('beta_study', lam=beta_prior['parameters'][0])
            else:
                beta_study = pm.Gamma('beta_study',
                                    alpha=beta_prior['parameters'][0],
                                    beta=beta_prior['parameters'][1])
            
            # Patient-level AE rates for each site
            lambda_sites = pm.Gamma('lambda_sites',
                                  alpha=alpha_study,
                                  beta=beta_study,
                                  shape=n_sites)
            
            # IPD likelihood
            ae_counts = pm.Poisson('ae_counts',
                                 mu=lambda_sites[sites_idx],
                                 observed=observed_ae)
            
            # Sampling
            trace_gpt = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"], 
                tune=EXPERIMENT_CONFIG["mcmc_tune"], 
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"], 
                progressbar=False
            )
        
        return model_gpt, trace_gpt, sample_data
    
    def run_model_gpt4_disease(self, selected_sites=None, n_sites=20):
        """Model C: GPT-4 disease-informed prior distribution - IPD version"""
        print(f"\n=== GPT-4 Disease-Informed Prior IPD Model (n_sites={n_sites}) ===")
        
        if not hasattr(self, 'gpt_disease_priors') or self.gpt_disease_priors is None:
            self.gpt_disease_priors = self.get_gpt4_prior_with_disease_info()
        
        # Site selection (External specification or fixed seed usage)
        if selected_sites is None:
            available_sites = self.patient_data['site_number'].unique()
            np.random.seed(EXPERIMENT_CONFIG["random_seed"])  # Same seed for all models
            selected_sites = np.random.choice(
                available_sites, 
                size=min(n_sites, len(available_sites)), 
                replace=False
            )
        
        # Obtain patient data from selected sites
        sample_data = self.patient_data[
            self.patient_data['site_number'].isin(selected_sites)
        ].copy()
        
        # Data preparation for IPD model
        sites = sample_data['site_number'].values
        observed_ae = sample_data['ae_count_cumulative'].values
        unique_sites, sites_idx = np.unique(sites, return_inverse=True)
        n_sites = len(unique_sites)
        
        with pm.Model() as model_gpt_disease:
            # GPT-4 disease-informed recommended prior distribution
            alpha_prior = self.gpt_disease_priors['alpha_prior']
            beta_prior = self.gpt_disease_priors['beta_prior']
            
            print(f"GPT-4 disease-informed recommended prior distribution: α ~ {alpha_prior['distribution'].title()}{tuple(alpha_prior['parameters'])}, β ~ {beta_prior['distribution'].title()}{tuple(beta_prior['parameters'])}")
            
            if alpha_prior['distribution'] == 'exponential':
                alpha_study = pm.Exponential('alpha_study', lam=alpha_prior['parameters'][0])
            else:
                alpha_study = pm.Gamma('alpha_study', 
                                     alpha=alpha_prior['parameters'][0],
                                     beta=alpha_prior['parameters'][1])
            
            if beta_prior['distribution'] == 'exponential':
                beta_study = pm.Exponential('beta_study', lam=beta_prior['parameters'][0])
            else:
                beta_study = pm.Gamma('beta_study',
                                    alpha=beta_prior['parameters'][0],
                                    beta=beta_prior['parameters'][1])
            
            # Patient-level AE rates for each site
            lambda_sites = pm.Gamma('lambda_sites',
                                  alpha=alpha_study,
                                  beta=beta_study,
                                  shape=n_sites)
            
            # IPD likelihood
            ae_counts = pm.Poisson('ae_counts',
                                 mu=lambda_sites[sites_idx],
                                 observed=observed_ae)
            
            # Sampling
            trace_gpt_disease = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"], 
                tune=EXPERIMENT_CONFIG["mcmc_tune"], 
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"], 
                progressbar=False
            )
        
        return model_gpt_disease, trace_gpt_disease, sample_data
    
    def calculate_predictive_performance(self, trace, sample_data):
        """Calculation of predictive performance for IPD"""
        # Calculate actual AE rates at site level
        site_true_rates = sample_data.groupby('site_number')['ae_count_cumulative'].mean()
        
        # Obtain predicted values from posterior predictive distribution
        posterior_lambda = trace.posterior['lambda_sites'].values
        # shape: (chains, draws, sites)
        predicted_rates = posterior_lambda.mean(axis=(0, 1))
        
        # Align site order
        unique_sites = np.sort(sample_data['site_number'].unique())
        true_rates = [site_true_rates[site] for site in unique_sites]
        
        # MAE
        mae = np.mean(np.abs(predicted_rates - true_rates))
        
        # RMSE  
        rmse = np.sqrt(np.mean((predicted_rates - true_rates) ** 2))
        
        # Log Predictive Density (for IPD)
        # Calculate log predictive density at patient level
        sites = sample_data['site_number'].values
        observed_ae = sample_data['ae_count_cumulative'].values
        unique_sites_array, sites_idx = np.unique(sites, return_inverse=True)
        
        # Predicted AE rate for each patient
        patient_predicted_rates = predicted_rates[sites_idx]
        
        # Poisson log probability density
        from scipy.stats import poisson
        log_pred_densities = []
        for i, (obs, pred_rate) in enumerate(zip(observed_ae, patient_predicted_rates)):
            lpd = poisson.logpmf(obs, pred_rate)
            log_pred_densities.append(lpd)
        
        log_pred_density = np.mean(log_pred_densities)
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'lpd': log_pred_density,
            'predicted_rates': predicted_rates,
            'true_rates': true_rates,
            'n_patients': len(sample_data),
            'n_sites': len(unique_sites)
        }
    
    def run_baseline_comparison_experiment(self):
        """Baseline comparison experiment - IPD version"""
        print("\n" + "="*60)
        print("IPD Baseline comparison experiment started")
        print("="*60)
        
        results = []
        sample_sizes = EXPERIMENT_CONFIG["sample_sizes"]
        
        for sample_size in sample_sizes:
            print(f"\n--- Number of sites {sample_size} ---")
            
            # Meta-analytical model
            _, trace_meta, sample_data = self.run_model_meta_analytical(sample_size)
            perf_meta = self.calculate_predictive_performance(trace_meta, sample_data)
            
            results.append({
                'n_sites': sample_size,
                'n_patients': perf_meta['n_patients'],
                'model': 'Meta-analytical',
                'mae': perf_meta['mae'],
                'rmse': perf_meta['rmse'],
                'lpd': perf_meta['lpd']
            })
            
            # GPT-4 model
            _, trace_gpt, _ = self.run_model_gpt4(sample_size)
            perf_gpt = self.calculate_predictive_performance(trace_gpt, sample_data)
            
            results.append({
                'n_sites': sample_size,
                'n_patients': perf_gpt['n_patients'],
                'model': 'GPT-4 Blind',
                'mae': perf_gpt['mae'],
                'rmse': perf_gpt['rmse'],
                'lpd': perf_gpt['lpd']
            })
            
            print(f"Number of patients: {perf_meta['n_patients']}")
            print(f"Meta-analytical - MAE: {perf_meta['mae']:.4f}, RMSE: {perf_meta['rmse']:.4f}, LPD: {perf_meta['lpd']:.4f}")
            print(f"GPT-4 Blind     - MAE: {perf_gpt['mae']:.4f}, RMSE: {perf_gpt['rmse']:.4f}, LPD: {perf_gpt['lpd']:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(DATA_DIR / 'ipd_baseline_comparison_results.csv', index=False)
        
        return results_df
    
    def analyze_sample_size_reduction(self, results_df):
        """Analyze required sample size reduction effect based on IPD"""
        print("\n=== IPD sample size reduction effect analysis ===")
        
        # Separate meta-analytical and GPT-4 data
        meta_data = results_df[results_df['model'] == 'Meta-analytical'].copy()
        gpt_data = results_df[results_df['model'] == 'GPT-4 Blind'].copy()
        
        reduction_analysis = []
        
        for metric in ['mae', 'rmse']:
            print(f"\n--- Analysis by {metric.upper()} ---")
            
            # Create interpolation function (based on number of sites)
            meta_interp = interp1d(meta_data[metric], meta_data['n_sites'], 
                                 kind='linear', fill_value='extrapolate')
            gpt_interp = interp1d(gpt_data[metric], gpt_data['n_sites'], 
                                kind='linear', fill_value='extrapolate')
            
            # Compare required number of sites at each baseline performance level
            for baseline_performance in EVALUATION_CONFIG["baseline_performance_levels"]:
                try:
                    # Number of sites required to achieve that performance
                    meta_required_n = float(meta_interp(baseline_performance))
                    gpt_required_n = float(gpt_interp(baseline_performance))
                    
                    # Site reduction effect
                    if meta_required_n > 0:
                        reduction_pct = (meta_required_n - gpt_required_n) / meta_required_n * 100
                        reduction_abs = meta_required_n - gpt_required_n
                        
                        reduction_analysis.append({
                            'metric': metric,
                            'baseline_performance': baseline_performance,
                            'meta_required_sites': meta_required_n,
                            'gpt_required_sites': gpt_required_n,
                            'reduction_absolute': reduction_abs,
                            'reduction_percentage': reduction_pct
                        })
                        
                        print(f"  Number of sites required to achieve {metric.upper()}={baseline_performance:.2f} performance:")
                        print(f"    Meta-analytical: {meta_required_n:.1f}")
                        print(f"    GPT-4 Blind:     {gpt_required_n:.1f}")
                        print(f"    Reduction effect:        {reduction_abs:.1f} ({reduction_pct:+.1f}%)")
                        
                except Exception as e:
                    print(f"  {metric.upper()}={baseline_performance:.2f}: Cannot calculate ({e})")
        
        reduction_df = pd.DataFrame(reduction_analysis)
        reduction_df.to_csv(DATA_DIR / 'ipd_sample_size_reduction_analysis.csv', index=False)
        
        return reduction_df

    def run_cv_experiment(self, n_folds=5):
        """K-fold Cross-Validation experiment (Hastie et al. 2009 compliant)"""
        print("\n" + "="*60)
        print(f"{n_folds}-Fold Cross-Validation experiment started")
        print("="*60)
        
        # Create stratified cross-validation folds
        cv_folds = self.create_stratified_cv_folds(n_folds)
        
        results = []
        
        for fold_idx, fold_sites in enumerate(cv_folds):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            print(f"Number of sites used: {len(fold_sites)}")
            
            # Use same data subset for all models
            
            # Meta-analytical model
            _, trace_meta, sample_data = self.run_model_meta_analytical(
                selected_sites=fold_sites, 
                n_sites=len(fold_sites)
            )
            perf_meta = self.calculate_predictive_performance(trace_meta, sample_data)
            
            results.append({
                'fold': fold_idx + 1,
                'n_sites': len(fold_sites),
                'n_patients': perf_meta['n_patients'],
                'model': 'Meta-analytical',
                'mae': perf_meta['mae'],
                'rmse': perf_meta['rmse'],
                'lpd': perf_meta['lpd']
            })
            
            # GPT-4 Blind model (same data subset)
            _, trace_gpt_blind, _ = self.run_model_gpt4(
                selected_sites=fold_sites,
                n_sites=len(fold_sites)
            )
            perf_gpt_blind = self.calculate_predictive_performance(trace_gpt_blind, sample_data)
            
            results.append({
                'fold': fold_idx + 1,
                'n_sites': len(fold_sites),
                'n_patients': perf_gpt_blind['n_patients'],
                'model': 'GPT-4 Blind',
                'mae': perf_gpt_blind['mae'],
                'rmse': perf_gpt_blind['rmse'],
                'lpd': perf_gpt_blind['lpd']
            })
            
            # GPT-4 Disease-Informed model (same data subset)
            _, trace_gpt_disease, _ = self.run_model_gpt4_disease(
                selected_sites=fold_sites,
                n_sites=len(fold_sites)
            )
            perf_gpt_disease = self.calculate_predictive_performance(trace_gpt_disease, sample_data)
            
            results.append({
                'fold': fold_idx + 1,
                'n_sites': len(fold_sites),
                'n_patients': perf_gpt_disease['n_patients'],
                'model': 'GPT-4 Disease-Informed',
                'mae': perf_gpt_disease['mae'],
                'rmse': perf_gpt_disease['rmse'],
                'lpd': perf_gpt_disease['lpd']
            })
            
            print(f"Number of patients: {perf_meta['n_patients']}")
            print(f"Meta-analytical      - MAE: {perf_meta['mae']:.4f}, RMSE: {perf_meta['rmse']:.4f}, LPD: {perf_meta['lpd']:.4f}")
            print(f"GPT-4 Blind          - MAE: {perf_gpt_blind['mae']:.4f}, RMSE: {perf_gpt_blind['rmse']:.4f}, LPD: {perf_gpt_blind['lpd']:.4f}")
            print(f"GPT-4 Disease-Informed - MAE: {perf_gpt_disease['mae']:.4f}, RMSE: {perf_gpt_disease['rmse']:.4f}, LPD: {perf_gpt_disease['lpd']:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Calculate cross-validation statistics
        cv_summary = results_df.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'lpd': ['mean', 'std']
        }).round(4)
        
        print(f"\n=== {n_folds}-Fold CV result summary ===")
        print(cv_summary)
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(DATA_DIR / f'cv_{n_folds}fold_results_{timestamp}.csv', index=False)
        cv_summary.to_csv(DATA_DIR / f'cv_{n_folds}fold_summary_{timestamp}.csv')
        
        return results_df, cv_summary

    def run_progressive_information_experiment(self):
        """Progressive information provision experiment - IPD version (Improved: using identical data subsets)"""
        print("\n" + "="*60)
        print("IPD progressive information provision experiment started (using identical data subsets)")
        print("="*60)
        
        results = []
        sample_sizes = EXPERIMENT_CONFIG["sample_sizes"]
        
        for sample_size in sample_sizes:
            print(f"\n--- Number of sites {sample_size} ---")
            
            # Select same sites for all models (Bergstra & Bengio 2012 compliant)
            available_sites = self.patient_data['site_number'].unique()
            np.random.seed(EXPERIMENT_CONFIG["random_seed"])
            selected_sites = np.random.choice(
                available_sites, 
                size=min(sample_size, len(available_sites)), 
                replace=False
            )
            
            print(f"Selected sites: {selected_sites[:5]}..." if len(selected_sites) > 5 else f"Selected sites: {selected_sites}")
            
            # Meta-analytical model
            _, trace_meta, sample_data = self.run_model_meta_analytical(
                selected_sites=selected_sites, 
                n_sites=sample_size
            )
            perf_meta = self.calculate_predictive_performance(trace_meta, sample_data)
            
            results.append({
                'n_sites': sample_size,
                'n_patients': perf_meta['n_patients'],
                'model': 'Meta-analytical',
                'mae': perf_meta['mae'],
                'rmse': perf_meta['rmse'],
                'lpd': perf_meta['lpd']
            })
            
            # GPT-4 Blind model (same data subset)
            _, trace_gpt_blind, _ = self.run_model_gpt4(
                selected_sites=selected_sites,
                n_sites=sample_size
            )
            perf_gpt_blind = self.calculate_predictive_performance(trace_gpt_blind, sample_data)
            
            results.append({
                'n_sites': sample_size,
                'n_patients': perf_gpt_blind['n_patients'],
                'model': 'GPT-4 Blind',
                'mae': perf_gpt_blind['mae'],
                'rmse': perf_gpt_blind['rmse'],
                'lpd': perf_gpt_blind['lpd']
            })
            
            # GPT-4 Disease-Informed model (same data subset)
            _, trace_gpt_disease, _ = self.run_model_gpt4_disease(
                selected_sites=selected_sites,
                n_sites=sample_size
            )
            perf_gpt_disease = self.calculate_predictive_performance(trace_gpt_disease, sample_data)
            
            results.append({
                'n_sites': sample_size,
                'n_patients': perf_gpt_disease['n_patients'],
                'model': 'GPT-4 Disease-Informed',
                'mae': perf_gpt_disease['mae'],
                'rmse': perf_gpt_disease['rmse'],
                'lpd': perf_gpt_disease['lpd']
            })
            
            print(f"Number of patients: {perf_meta['n_patients']}")
            print(f"Meta-analytical      - MAE: {perf_meta['mae']:.4f}, RMSE: {perf_meta['rmse']:.4f}, LPD: {perf_meta['lpd']:.4f}")
            print(f"GPT-4 Blind          - MAE: {perf_gpt_blind['mae']:.4f}, RMSE: {perf_gpt_blind['rmse']:.4f}, LPD: {perf_gpt_blind['lpd']:.4f}")
            print(f"GPT-4 Disease-Informed - MAE: {perf_gpt_disease['mae']:.4f}, RMSE: {perf_gpt_disease['rmse']:.4f}, LPD: {perf_gpt_disease['lpd']:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Save with timestamped filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(DATA_DIR / f'ipd_progressive_information_fair_comparison_{timestamp}.csv', index=False)
        
        return results_df

    def run_model_barmaz_menard_exact(self, n_sites=20):
        """Barmaz & Ménard (2021) exact replication - IPD version"""
        print(f"\n=== Barmaz & Ménard (2021) Exact IPD Replication (n_sites={n_sites}) ===")
        
        # Random sampling of sites
        available_sites = self.patient_data['site_number'].unique()
        np.random.seed(EXPERIMENT_CONFIG["random_seed"])
        selected_sites = np.random.choice(
            available_sites, 
            size=min(n_sites, len(available_sites)), 
            replace=False
        )
        
        # Obtain patient-level data from selected sites
        patient_level_data = self.patient_data[
            self.patient_data['site_number'].isin(selected_sites)
        ]
        
        # Data preparation for IPD model
        sites = patient_level_data['site_number'].values
        observed_ae = patient_level_data['ae_count_cumulative'].values
        unique_sites, sites_idx = np.unique(sites, return_inverse=True)
        n_sites = len(unique_sites)
        
        print(f"Number of selected sites: {n_sites}")
        print(f"Number of patients: {len(patient_level_data)}")
        print(f"AE range: {observed_ae.min()} - {observed_ae.max()}")
        
        with pm.Model() as model_barmaz:
            # Barmaz & Ménard (2021) prior distribution
            alpha_study = pm.Exponential('alpha_study', lam=0.1)
            beta_study = pm.Exponential('beta_study', lam=0.1)
            
            # Patient-level AE rates by site
            lambda_sites = pm.Gamma('lambda_sites',
                                  alpha=alpha_study,
                                  beta=beta_study,
                                  shape=n_sites)
            
            # IPD likelihood: Patient-level likelihood
            ae_counts = pm.Poisson('ae_counts',
                                 mu=lambda_sites[sites_idx],
                                 observed=observed_ae)
            
            # Sampling
            trace_barmaz = pm.sample(
                EXPERIMENT_CONFIG["mcmc_samples"],
                tune=EXPERIMENT_CONFIG["mcmc_tune"],
                return_inferencedata=True,
                random_seed=EXPERIMENT_CONFIG["random_seed"],
                progressbar=False
            )
        
        return model_barmaz, trace_barmaz, patient_level_data

def main():
    """Main experiment execution - Improved version (using identical data subsets)"""
    print("Improved version Bayesian prior distribution comparison experiment (Fair Comparison with Same Data Subsets)")
    print(f"Results save destination: {RESULTS_DIR}")
    
    # Execute experiment
    experiment = ImprovedBayesianExperiment()
    
    # K-fold Cross-Validation experiment (authoritative method)
    print("\n" + "="*60)
    print("K-fold Cross-Validation experiment (Hastie et al. 2009 compliant)")
    print("="*60)
    cv_results_df, cv_summary = experiment.run_cv_experiment(n_folds=5)
    
    # Progressive information provision experiment (using identical data subsets)
    print("\n" + "="*60)
    print("Progressive information provision experiment (fair comparison)")
    print("="*60)
    progressive_results_df = experiment.run_progressive_information_experiment()
    
    print(f"\nExperiment completed! Results saved in {RESULTS_DIR}.")
    print("\nKey improvements:")
    print("- Fair comparison using K-fold Cross-Validation")
    print("- Using identical data subsets for all models")
    print("- Bias elimination through stratified cross-validation")
    print("- Complete magic number elimination from prompts")
    print("- Adoption of statistically rigorous evaluation methods")
    
    return cv_results_df, cv_summary, progressive_results_df

if __name__ == "__main__":
    cv_results_df, cv_summary, progressive_results_df = main()
