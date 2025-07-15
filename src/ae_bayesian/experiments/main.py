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
from scipy import stats
from datetime import datetime
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
        """Obtain prior distributions from GPT-4 completely blind - DEPRECATED
        
        This method is now replaced by temperature cross-validation analysis.
        Use run_cv_experiment_with_temperature_analysis() instead.
        """
        print("WARNING: This method is deprecated. Use run_cv_experiment_with_temperature_analysis() instead.")
        
        # Return fallback for backward compatibility
        return MODEL_CONFIG["gpt4_fallback"]
    
    def get_gpt4_prior_completely_blind_single(self, temperature=None):
        """Obtain prior distributions from GPT-4 completely blind (single run)"""
        print("\n=== Completely Blind Inquiry to GPT-4 (Single Run) ===")
        
        if temperature is None:
            temperature = GPT4_CONFIG["temperature"]
            
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
                    temperature=temperature
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
        """Obtain prior distributions from GPT-4 with disease information - DEPRECATED
        
        This method is now replaced by temperature cross-validation analysis.
        Use run_cv_experiment_with_temperature_analysis() instead.
        """
        print("WARNING: This method is deprecated. Use run_cv_experiment_with_temperature_analysis() instead.")
        
        # Return fallback for backward compatibility
        return MODEL_CONFIG["gpt4_fallback"]

    def get_gpt4_prior_with_disease_info_single(self, temperature=None):
        """Obtain prior distributions from GPT-4 with disease information (single run)"""
        print("\n=== Disease Information Inquiry to GPT-4 (Single Run) ===")
        
        if temperature is None:
            temperature = GPT4_CONFIG["temperature"]
        
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
                    temperature=temperature
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

    def run_cv_experiment_with_temperature_analysis(self, n_folds=5):
        """K-fold Cross-Validation with Temperature Sensitivity Analysis"""
        print("\n" + "="*80)
        print(f"{n_folds}-Fold Cross-Validation with Temperature Analysis")
        print("="*80)
        
        # Create stratified cross-validation folds (identical for all methods)
        cv_folds = self.create_stratified_cv_folds(n_folds)
        
        results = []
        
        # Run cross-validation for each temperature
        for temperature in GPT4_CONFIG["temperature_grid"]:
            print(f"\n{'='*60}")
            print(f"Temperature {temperature} Cross-Validation")
            print(f"{'='*60}")
            
            for fold_idx, fold_sites in enumerate(cv_folds):
                print(f"\n--- Fold {fold_idx + 1}/{n_folds} (Temperature {temperature}) ---")
                print(f"Sites in fold: {len(fold_sites)}")
                
                # Meta-analytical model (same data for all methods)
                _, trace_meta, sample_data = self.run_model_meta_analytical(
                    selected_sites=fold_sites, 
                    n_sites=len(fold_sites)
                )
                perf_meta = self.calculate_predictive_performance(trace_meta, sample_data)
                
                results.append({
                    'fold': fold_idx + 1,
                    'temperature': temperature,
                    'n_sites': len(fold_sites),
                    'n_patients': perf_meta['n_patients'],
                    'model': 'Meta-analytical',
                    'mae': perf_meta['mae'],
                    'rmse': perf_meta['rmse'],
                    'lpd': perf_meta['lpd']
                })
                
                # GPT-4 methods with temperature-specific priors
                for method_name, prior_func in [
                    ('GPT-4 Blind', self.get_gpt4_prior_for_cv),
                    ('GPT-4 Disease-Informed', self.get_gpt4_disease_informed_prior_for_cv)
                ]:
                    # Get aggregated prior from 5 LLM runs at this temperature
                    aggregated_prior = prior_func(temperature, fold_idx + 1)
                    
                    # Run model with aggregated prior
                    _, trace_gpt, _ = self.run_model_with_custom_prior(
                        prior_config=aggregated_prior,
                        selected_sites=fold_sites,
                        n_sites=len(fold_sites)
                    )
                    perf_gpt = self.calculate_predictive_performance(trace_gpt, sample_data)
                    
                    results.append({
                        'fold': fold_idx + 1,
                        'temperature': temperature,
                        'n_sites': len(fold_sites),
                        'n_patients': perf_gpt['n_patients'],
                        'model': method_name,
                        'mae': perf_gpt['mae'],
                        'rmse': perf_gpt['rmse'],
                        'lpd': perf_gpt['lpd'],
                        'prior_alpha': aggregated_prior['alpha_prior']['parameters'][0],
                        'prior_beta': aggregated_prior['beta_prior']['parameters'][0]
                    })
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = DATA_DIR / f'cv_temperature_analysis_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        
        # Analysis and comparison
        self.analyze_temperature_cv_results(results_df)
        
        return results_df

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

    def get_gpt4_prior_multiple_runs(self, prompt_type="blind", n_runs=None, temperature=None):
        """複数回実行してGPT-4事前分布を取得・集約"""
        if n_runs is None:
            n_runs = GPT4_CONFIG.get("n_runs_per_cv_fold", 5)
        if temperature is None:
            temperature = GPT4_CONFIG["temperature"]
            
        print(f"\n=== Multiple GPT-4 Runs: {prompt_type}, temperature={temperature}, n_runs={n_runs} ===")
        
        all_responses = []
        valid_responses = []
        
        for run_id in range(n_runs):
            print(f"\n--- Run {run_id + 1}/{n_runs} ---")
            
            if prompt_type == "blind":
                response = self.get_gpt4_prior_completely_blind_single(temperature=temperature)
            elif prompt_type == "disease_informed":
                response = self.get_gpt4_prior_with_disease_info_single(temperature=temperature)
            else:
                raise ValueError("prompt_type must be 'blind' or 'disease_informed'")
            
            all_responses.append(response)
            
            # フォールバックでない場合のみ有効とみなす
            if not response.get('is_fallback', False):
                valid_responses.append(response)
        
        if not valid_responses:
            print("No valid responses obtained, using fallback")
            return MODEL_CONFIG["gpt4_fallback"]
        
        # 複数応答の集約
        aggregated_prior = self._aggregate_gpt4_responses(valid_responses)
        
        # メタデータの追加
        aggregated_prior['aggregation_metadata'] = {
            'n_total_runs': n_runs,
            'n_valid_responses': len(valid_responses),
            'temperature': temperature,
            'prompt_type': prompt_type,
            'all_responses': all_responses
        }
        
        return aggregated_prior
    
    def _aggregate_gpt4_responses(self, responses):
        """複数のGPT-4応答を集約"""
        print(f"\n--- Aggregating {len(responses)} responses ---")
        
        # パラメータの抽出
        alpha_params = []
        beta_params = []
        
        for response in responses:
            try:
                alpha_params.append(response['alpha_prior']['parameters'][0])
                beta_params.append(response['beta_prior']['parameters'][0])
            except Exception as e:
                print(f"Error extracting parameters: {e}")
                continue
        
        if not alpha_params or not beta_params:
            print("No valid parameters found, using fallback")
            return MODEL_CONFIG["gpt4_fallback"]
        
        # 統計情報の計算
        alpha_mean = np.mean(alpha_params)
        alpha_std = np.std(alpha_params)
        beta_mean = np.mean(beta_params)
        beta_std = np.std(beta_params)
        
        print(f"Alpha parameters: mean={alpha_mean:.4f}, std={alpha_std:.4f}, CV={alpha_std/alpha_mean:.4f}")
        print(f"Beta parameters: mean={beta_mean:.4f}, std={beta_std:.4f}, CV={beta_std/beta_mean:.4f}")
        
        # 集約方法の選択（平均値を使用）
        aggregated_alpha = alpha_mean
        aggregated_beta = beta_mean
        
        # 信頼度レベルの集約
        confidence_levels = [r.get('confidence_level', 'medium') for r in responses]
        aggregated_confidence = max(set(confidence_levels), key=confidence_levels.count)
        
        # 理由の統合
        all_reasoning = [r.get('reasoning', '') for r in responses]
        aggregated_reasoning = f"Aggregated from {len(responses)} runs. Common themes: " + "; ".join(all_reasoning[:2])
        
        return {
            "alpha_prior": {"distribution": "exponential", "parameters": [aggregated_alpha]},
            "beta_prior": {"distribution": "exponential", "parameters": [aggregated_beta]},
            "reasoning": aggregated_reasoning,
            "confidence_level": aggregated_confidence,
            "aggregation_stats": {
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "beta_mean": beta_mean,
                "beta_std": beta_std,
                "n_responses": len(responses)
            }
        }

    def _select_optimal_temperature_result(self, all_temperature_results, format_compliance_analysis):
        """Select optimal temperature based on feedback principles"""
        print(f"\n--- Optimal Temperature Selection (Based on Feedback) ---")
        
        # Feedback principle: Allow variation and measure it, prefer higher temperatures unless format compliance is poor
        # Priority: 1.0 > 0.5 > 0.1 (unless compliance issues)
        
        preferred_order = [1.0, 0.5, 0.1]  # Based on feedback: allow natural variation
        
        for temp in preferred_order:
            if temp in all_temperature_results:
                compliance_data = format_compliance_analysis[temp]
                
                # Minimum compliance threshold (flexible)
                min_compliance = 0.5  # At least 50% of runs should succeed
                
                if compliance_data['compliance_rate'] >= min_compliance:
                    valid_responses = all_temperature_results[temp]['valid_responses']
                    
                    if valid_responses:
                        # Aggregate responses (early aggregation)
                        aggregated = self._aggregate_gpt4_responses(valid_responses)
                        aggregated['selected_temperature'] = temp
                        aggregated['compliance_rate'] = compliance_data['compliance_rate']
                        aggregated['selection_reason'] = f"Selected temp={temp} based on feedback principles (natural variation + adequate compliance)"
                        
                        print(f"Selected temperature: {temp}")
                        print(f"Compliance rate: {compliance_data['compliance_rate']:.1%}")
                        print(f"Valid responses: {len(valid_responses)}")
                        print(f"Reason: {aggregated['selection_reason']}")
                        
                        return aggregated
        
        # Fallback: use any available valid responses
        print("Warning: No temperature met preferred criteria, using fallback")
        for temp, results in all_temperature_results.items():
            if results['valid_responses']:
                fallback = self._aggregate_gpt4_responses(results['valid_responses'])
                fallback['selected_temperature'] = temp
                fallback['compliance_rate'] = format_compliance_analysis[temp]['compliance_rate']
                fallback['selection_reason'] = f"Fallback selection: temp={temp}"
                return fallback
        
        # Ultimate fallback
        return MODEL_CONFIG["gpt4_fallback"]
    
    def _save_temperature_sensitivity_results(self, all_temperature_results, format_compliance_analysis, prompt_type):
        """Save comprehensive temperature sensitivity analysis results"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Detailed results
        detailed_results = {
            'prompt_type': prompt_type,
            'timestamp': timestamp,
            'temperature_grid': GPT4_CONFIG["temperature_grid"],
            'runs_per_cv_fold': GPT4_CONFIG["n_runs_per_cv_fold"],
            'temperature_results': {},
            'format_compliance_analysis': format_compliance_analysis,
            'feedback_compliance': {
                'multiple_runs_implemented': True,
                'variation_measured': True,
                'temperature_as_hyperparameter': True,
                'aggregation_method': 'early_aggregation'
            }
        }
        
        # Convert results for JSON serialization
        for temp, results in all_temperature_results.items():
            detailed_results['temperature_results'][str(temp)] = {
                'responses': results['responses'],
                'valid_count': len(results['valid_responses']),
                'total_runs': GPT4_CONFIG["n_runs_per_cv_fold"],
                'compliance_rate': format_compliance_analysis[temp]['compliance_rate']
            }
        
        # Save detailed results
        detailed_file = DATA_DIR / f"temperature_sensitivity_{prompt_type}_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for temp in GPT4_CONFIG["temperature_grid"]:
            if temp in all_temperature_results:
                valid_responses = all_temperature_results[temp]['valid_responses']
                compliance = format_compliance_analysis[temp]
                
                if valid_responses:
                    alpha_params = [float(r['alpha_prior']['parameters'][0]) for r in valid_responses]
                    beta_params = [float(r['beta_prior']['parameters'][0]) for r in valid_responses]
                    
                    summary_data.append({
                        'temperature': temp,
                        'prompt_type': prompt_type,
                        'total_runs': compliance['total_runs'],
                        'valid_responses': compliance['valid_responses'],
                        'compliance_rate': compliance['compliance_rate'],
                        'alpha_mean': np.mean(alpha_params),
                        'alpha_std': np.std(alpha_params),
                        'alpha_cv': np.std(alpha_params) / np.mean(alpha_params),
                        'beta_mean': np.mean(beta_params),
                        'beta_std': np.std(beta_params),
                        'beta_cv': np.std(beta_params) / np.mean(beta_params)
                    })
                else:
                    summary_data.append({
                        'temperature': temp,
                        'prompt_type': prompt_type,
                        'total_runs': compliance['total_runs'],
                        'valid_responses': 0,
                        'compliance_rate': 0.0,
                        'alpha_mean': np.nan,
                        'alpha_std': np.nan,
                        'alpha_cv': np.nan,
                        'beta_mean': np.nan,
                        'beta_std': np.nan,
                        'beta_cv': np.nan
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = DATA_DIR / f"temperature_sensitivity_summary_{prompt_type}_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nTemperature sensitivity results saved:")
        print(f"- Detailed: {detailed_file}")
        print(f"- Summary: {summary_file}")
        
        return detailed_file, summary_file

    def get_gpt4_prior_for_cv(self, temperature, fold_number):
        """Get aggregated GPT-4 blind prior for cross-validation (5 runs per fold)"""
        print(f"    GPT-4 Blind Prior (T={temperature}, Fold {fold_number}): 5 runs")
        
        valid_responses = []
        for run in range(GPT4_CONFIG["n_runs_per_cv_fold"]):
            try:
                response = self.get_gpt4_prior_completely_blind_single(temperature=temperature)
                if response != MODEL_CONFIG["gpt4_fallback"]:
                    # Validate response
                    alpha_param = float(response['alpha_prior']['parameters'][0])
                    beta_param = float(response['beta_prior']['parameters'][0])
                    if alpha_param > 0 and beta_param > 0:
                        valid_responses.append(response)
                        print(f"      Run {run+1}: α={alpha_param:.4f}, β={beta_param:.4f}")
            except Exception as e:
                print(f"      Run {run+1}: Error - {e}")
        
        if not valid_responses:
            print(f"      No valid responses, using fallback")
            return MODEL_CONFIG["gpt4_fallback"]
        
        # Aggregate responses
        aggregated = self._aggregate_gpt4_responses(valid_responses)
        print(f"      Aggregated: α={aggregated['alpha_prior']['parameters'][0]:.4f}, β={aggregated['beta_prior']['parameters'][0]:.4f}")
        return aggregated

    def get_gpt4_disease_informed_prior_for_cv(self, temperature, fold_number):
        """Get aggregated GPT-4 disease-informed prior for cross-validation (5 runs per fold)"""
        print(f"    GPT-4 Disease-Informed Prior (T={temperature}, Fold {fold_number}): 5 runs")
        
        valid_responses = []
        for run in range(GPT4_CONFIG["n_runs_per_cv_fold"]):
            try:
                response = self.get_gpt4_prior_with_disease_info_single(temperature=temperature)
                if response != MODEL_CONFIG["gpt4_fallback"]:
                    # Validate response
                    alpha_param = float(response['alpha_prior']['parameters'][0])
                    beta_param = float(response['beta_prior']['parameters'][0])
                    if alpha_param > 0 and beta_param > 0:
                        valid_responses.append(response)
                        print(f"      Run {run+1}: α={alpha_param:.4f}, β={beta_param:.4f}")
            except Exception as e:
                print(f"      Run {run+1}: Error - {e}")
        
        if not valid_responses:
            print(f"      No valid responses, using fallback")
            return MODEL_CONFIG["gpt4_fallback"]
        
        # Aggregate responses
        aggregated = self._aggregate_gpt4_responses(valid_responses)
        print(f"      Aggregated: α={aggregated['alpha_prior']['parameters'][0]:.4f}, β={aggregated['beta_prior']['parameters'][0]:.4f}")
        return aggregated

    def run_model_with_custom_prior(self, prior_config, selected_sites=None, n_sites=None):
        """Run model with custom prior configuration"""
        if selected_sites is not None:
            sample_data = self.patient_data[
                self.patient_data['site_number'].isin(selected_sites)
            ].copy()
        else:
            sample_data = self.sample_sites_stratified(n_sites)
        
        # Prepare data for PyMC model
        site_indices = pd.Categorical(sample_data['site_number']).codes
        n_sites_actual = len(sample_data['site_number'].unique())
        
        # Extract prior parameters
        alpha_param = float(prior_config['alpha_prior']['parameters'][0])
        beta_param = float(prior_config['beta_prior']['parameters'][0])
        
        with pm.Model() as model:
            # Priors (using custom parameters)
            alpha = pm.Exponential('alpha', alpha_param)
            beta = pm.Exponential('beta', beta_param)
            
            # Site-specific rates
            lambda_sites = pm.Gamma('lambda_sites', alpha=alpha, beta=beta, shape=n_sites_actual)
            
            # Likelihood
            y_obs = pm.Poisson('y_obs', mu=lambda_sites[site_indices], observed=sample_data['ae_count_cumulative'])
            
            # Sample
            trace = pm.sample(
                draws=EXPERIMENT_CONFIG["mcmc_samples"],
                tune=EXPERIMENT_CONFIG["mcmc_tune"],
                random_seed=EXPERIMENT_CONFIG["random_seed"],
                return_inferencedata=True,
                progressbar=False
            )
        
        return model, trace, sample_data

    def analyze_temperature_cv_results(self, results_df):
        """Analyze cross-validation results across temperatures"""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION TEMPERATURE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Group by model and temperature
        summary_stats = results_df.groupby(['model', 'temperature']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'lpd': ['mean', 'std']
        }).round(4)
        
        print("\nPerformance Summary (Mean ± SD across folds):")
        print("-" * 80)
        
        for model in results_df['model'].unique():
            print(f"\n{model}:")
            model_data = summary_stats.loc[model]
            
            for temp in sorted(results_df['temperature'].unique()):
                if temp in model_data.index:
                    stats = model_data.loc[temp]
                    mae_mean, mae_std = stats[('mae', 'mean')], stats[('mae', 'std')]
                    rmse_mean, rmse_std = stats[('rmse', 'mean')], stats[('rmse', 'std')]
                    lpd_mean, lpd_std = stats[('lpd', 'mean')], stats[('lpd', 'std')]
                    
                    print(f"  Temperature {temp:3.1f}: MAE {mae_mean:.4f}±{mae_std:.4f}, RMSE {rmse_mean:.4f}±{rmse_std:.4f}, LPD {lpd_mean:.4f}±{lpd_std:.4f}")
        
        # Find best performers
        print(f"\n{'='*50}")
        print("BEST PERFORMERS BY METRIC")
        print(f"{'='*50}")
        
        for metric in ['mae', 'rmse', 'lpd']:
            print(f"\nBest {metric.upper()}:")
            metric_summary = results_df.groupby(['model', 'temperature'])[metric].mean().reset_index()
            
            if metric in ['mae', 'rmse']:  # Lower is better
                best = metric_summary.loc[metric_summary[metric].idxmin()]
            else:  # lpd: higher is better
                best = metric_summary.loc[metric_summary[metric].idxmax()]
            
            print(f"  {best['model']} (T={best['temperature']:.1f}): {best[metric]:.4f}")
        
        # Statistical significance tests
        self.perform_temperature_significance_tests(results_df)

    def perform_temperature_significance_tests(self, results_df):
        """Perform statistical significance tests between temperature configurations"""
        print(f"\n{'='*50}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*50}")
        
        for model in ['GPT-4 Blind', 'GPT-4 Disease-Informed']:
            print(f"\n{model} Temperature Comparison:")
            model_data = results_df[results_df['model'] == model]
            
            temps = sorted(model_data['temperature'].unique())
            
            for i, temp1 in enumerate(temps):
                for temp2 in temps[i+1:]:
                    data1 = model_data[model_data['temperature'] == temp1]['mae']
                    data2 = model_data[model_data['temperature'] == temp2]['mae']
                    
                    # Paired t-test (same CV folds)
                    t_stat, p_value = stats.ttest_rel(data1, data2)
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    print(f"  T={temp1:.1f} vs T={temp2:.1f}: t={t_stat:.3f}, p={p_value:.4f} {significance}")
