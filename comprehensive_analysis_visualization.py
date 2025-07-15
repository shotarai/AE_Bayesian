#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization for Temperature Sensitivity Study
=========================================================================

This script creates detailed visualizations and analysis addressing:
1. Box plots comparing performance across methods and temperatures
2. Prior parameter distributions and comparisons
3. Cross-validation fold analysis
4. Sample size and statistical power analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def load_and_prepare_data():
    """Load and prepare the temperature analysis data"""
    data_path = Path("results/data/cv_temperature_analysis_20250715_125755.csv")
    df = pd.read_csv(data_path)
    
    # Round to 3 significant figures for publication quality
    df['mae_rounded'] = df['mae'].round(3)
    df['rmse_rounded'] = df['rmse'].round(3)
    df['lpd_rounded'] = df['lpd'].round(2)
    
    return df

def create_individual_performance_boxplots(df):
    """Create individual box plots for each performance metric"""
    metrics = ['mae', 'rmse', 'lpd']
    metric_names = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'Log Predictive Density (LPD)']
    metric_units = ['', '', '']
    
    # Prepare data with proper labels
    plot_data = []
    for _, row in df.iterrows():
        if row['model'] == 'Meta-analytical':
            label = 'Meta-analytical'
            color_group = 'Meta-analytical'
        else:
            if 'Blind' in row['model']:
                method_type = 'Blind'
            else:
                method_type = 'Disease-Informed'
            label = f"GPT-4 {method_type}\n(T={row['temperature']:.1f})"
            color_group = f"{method_type}_T{row['temperature']:.1f}"
        
        plot_data.append({
            'mae': row['mae'],
            'rmse': row['rmse'], 
            'lpd': row['lpd'],
            'label': label,
            'color_group': color_group,
            'model_type': 'Meta-analytical' if row['model'] == 'Meta-analytical' else method_type,
            'temperature': row['temperature'] if row['model'] != 'Meta-analytical' else 'N/A'
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors
    colors = {
        'Meta-analytical': '#808080',  # Gray
        'Blind_T0.1': '#1f77b4',      # Blue
        'Blind_T0.5': '#2ca02c',      # Green  
        'Blind_T1.0': '#ff7f0e',      # Orange
        'Disease-Informed_T0.1': '#d62728',  # Red
        'Disease-Informed_T0.5': '#9467bd',  # Purple
        'Disease-Informed_T1.0': '#8c564b'   # Brown
    }
    
    for idx, (metric, name, unit) in enumerate(zip(metrics, metric_names, metric_units)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create box plot
        unique_labels = plot_df['label'].unique()
        positions = range(len(unique_labels))
        
        box_data = []
        box_colors = []
        labels = []
        
        for label in unique_labels:
            subset = plot_df[plot_df['label'] == label]
            box_data.append(subset[metric].values)
            box_colors.append(colors[subset['color_group'].iloc[0]])
            labels.append(label)
        
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, 
                       widths=0.6, showmeans=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize the plot
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'{name} {unit}'.strip())
        ax.set_title(f'{name} Performance Comparison\n(5-fold Cross-Validation, n=5 per condition)', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend outside the plot area
        legend_elements = []
        for color_group, color in colors.items():
            if color_group in plot_df['color_group'].values:
                if color_group == 'Meta-analytical':
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label='Meta-analytical'))
                else:
                    parts = color_group.split('_')
                    method = parts[0]
                    temp = parts[1]
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
                                                       label=f'GPT-4 {method} ({temp})'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')
        
        plt.tight_layout()
        
        # Save individual plot
        output_path = Path(f"results/plots/boxplot_{metric}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {output_path}")
    
    return True

def create_individual_prior_parameter_plots(df):
    """Create individual plots for prior parameter analysis"""
    # Filter GPT-4 data only
    gpt_data = df[df['model'].str.contains('GPT-4')].copy()
    
    # Prepare data with proper labels
    plot_data = []
    for _, row in gpt_data.iterrows():
        if 'Blind' in row['model']:
            method_type = 'Blind'
        else:
            method_type = 'Disease-Informed'
        
        label = f"GPT-4 {method_type}\n(T={row['temperature']:.1f})"
        
        plot_data.append({
            'prior_alpha': row['prior_alpha'],
            'prior_beta': row['prior_beta'],
            'label': label,
            'method_type': method_type,
            'temperature': row['temperature']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors for consistency
    colors = {
        'Blind_T0.1': '#1f77b4',      # Blue
        'Blind_T0.5': '#2ca02c',      # Green  
        'Blind_T1.0': '#ff7f0e',      # Orange
        'Disease-Informed_T0.1': '#d62728',  # Red
        'Disease-Informed_T0.5': '#9467bd',  # Purple
        'Disease-Informed_T1.0': '#8c564b'   # Brown
    }
    
    # 1. Alpha parameter boxplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    unique_labels = plot_df['label'].unique()
    positions = range(len(unique_labels))
    
    box_data = []
    box_colors = []
    labels = []
    
    for label in unique_labels:
        subset = plot_df[plot_df['label'] == label]
        box_data.append(subset['prior_alpha'].values)
        method = subset['method_type'].iloc[0]
        temp = subset['temperature'].iloc[0]
        color_key = f"{method}_T{temp:.1f}"
        box_colors.append(colors[color_key])
        labels.append(label)
    
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, 
                   widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Alpha (Exponential Rate Parameter)')
    ax.set_title('Alpha Parameter Distribution Across CV Folds\n(5-fold Cross-Validation, n=5 per condition)', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend outside the plot area
    legend_elements = []
    for method in ['Blind', 'Disease-Informed']:
        for temp in [0.1, 0.5, 1.0]:
            color_key = f"{method}_T{temp:.1f}"
            if color_key in colors:
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[color_key], alpha=0.7, 
                                                   label=f'GPT-4 {method} (T={temp:.1f})'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')
    
    plt.tight_layout()
    output_path = Path("results/plots/alpha_parameter_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    # 2. Beta parameter boxplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    box_data = []
    box_colors = []
    labels = []
    
    for label in unique_labels:
        subset = plot_df[plot_df['label'] == label]
        box_data.append(subset['prior_beta'].values)
        method = subset['method_type'].iloc[0]
        temp = subset['temperature'].iloc[0]
        color_key = f"{method}_T{temp:.1f}"
        box_colors.append(colors[color_key])
        labels.append(label)
    
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, 
                   widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Beta (Exponential Rate Parameter)')
    ax.set_title('Beta Parameter Distribution Across CV Folds\n(5-fold Cross-Validation, n=5 per condition)', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend outside the plot area
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')
    
    plt.tight_layout()
    output_path = Path("results/plots/beta_parameter_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    # 3. Parameter summary table as separate plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    # Create summary table
    summary_stats = []
    for method in ['Blind', 'Disease-Informed']:
        for temp in [0.1, 0.5, 1.0]:
            subset = gpt_data[(gpt_data['model'].str.contains(method)) & (gpt_data['temperature'] == temp)]
            if len(subset) > 0:
                alpha_mean = subset['prior_alpha'].mean()
                alpha_std = subset['prior_alpha'].std()
                beta_mean = subset['prior_beta'].mean()
                beta_std = subset['prior_beta'].std()
                
                summary_stats.append({
                    'Method': f'GPT-4 {method}',
                    'Temperature': f'{temp:.1f}',
                    'α (mean±SD)': f'{alpha_mean:.3f}±{alpha_std:.3f}',
                    'β (mean±SD)': f'{beta_mean:.3f}±{beta_std:.3f}',
                    'n (CV folds)': len(subset)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create table
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.0)
    
    # Style the table
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))
    
    # Color header row
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Prior Parameter Summary Statistics\n(5 CV folds per condition)', 
                fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    output_path = Path("results/plots/prior_parameter_summary_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")
    
    return True

# Remove unused functions - keeping only the essential ones for individual plots

def create_summary_table():
    """Create publication-ready summary table with appropriate precision"""
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate summary statistics with appropriate precision
    summary_stats = []
    
    for model in df['model'].unique():
        if 'Meta-analytical' in model:
            model_data = df[df['model'] == model]
            mae_mean = model_data['mae'].mean()
            mae_std = model_data['mae'].std()
            rmse_mean = model_data['rmse'].mean()
            rmse_std = model_data['rmse'].std()
            lpd_mean = model_data['lpd'].mean()
            lpd_std = model_data['lpd'].std()
            n_folds = len(model_data)
            
            summary_stats.append({
                'Method': 'Meta-analytical',
                'Temperature': 'N/A',
                'MAE': f'{mae_mean:.3f}±{mae_std:.3f}',
                'RMSE': f'{rmse_mean:.3f}±{rmse_std:.3f}',
                'LPD': f'{lpd_mean:.2f}±{lpd_std:.2f}',
                'n (CV folds)': n_folds,
                'n (Sites per fold)': f"{model_data['n_sites'].iloc[0]}",
                'n (Patients)': f"{model_data['n_patients'].sum()}"
            })
        else:
            for temp in df['temperature'].unique():
                model_temp_data = df[(df['model'] == model) & (df['temperature'] == temp)]
                if len(model_temp_data) > 0:
                    mae_mean = model_temp_data['mae'].mean()
                    mae_std = model_temp_data['mae'].std()
                    rmse_mean = model_temp_data['rmse'].mean()
                    rmse_std = model_temp_data['rmse'].std()
                    lpd_mean = model_temp_data['lpd'].mean()
                    lpd_std = model_temp_data['lpd'].std()
                    n_folds = len(model_temp_data)
                    
                    summary_stats.append({
                        'Method': model,
                        'Temperature': f'{temp:.1f}',
                        'MAE': f'{mae_mean:.3f}±{mae_std:.3f}',
                        'RMSE': f'{rmse_mean:.3f}±{rmse_std:.3f}',
                        'LPD': f'{lpd_mean:.2f}±{lpd_std:.2f}',
                        'n (CV folds)': n_folds,
                        'n (Sites per fold)': f"{model_temp_data['n_sites'].iloc[0]}",
                        'n (Patients)': f"{model_temp_data['n_patients'].sum()}"
                    })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by MAE performance
    summary_df['mae_numeric'] = summary_df['MAE'].str.split('±').str[0].astype(float)
    summary_df = summary_df.sort_values('mae_numeric').drop('mae_numeric', axis=1)
    
    # Save to CSV
    output_path = Path("results/data/publication_ready_summary_table.csv")
    summary_df.to_csv(output_path, index=False)
    
    print("Publication-Ready Summary Table:")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\n")
    print(f"Table saved to: {output_path}")
    
    return summary_df

def main():
    """Main execution function"""
    print("Creating Individual Analysis and Visualizations...")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create individual visualizations
    print("\n1. Creating individual performance box plots (MAE, RMSE, LPD)...")
    create_individual_performance_boxplots(df)
    
    print("\n2. Creating individual prior parameter plots (Alpha, Beta, Summary Table)...")
    create_individual_prior_parameter_plots(df)
    
    print("\n3. Creating publication-ready summary table...")
    create_summary_table()
    
    print("\n" + "="*60)
    print("All individual visualizations completed!")
    print("Generated files:")
    print("- results/plots/boxplot_mae.png")
    print("- results/plots/boxplot_rmse.png") 
    print("- results/plots/boxplot_lpd.png")
    print("- results/plots/alpha_parameter_boxplot.png")
    print("- results/plots/beta_parameter_boxplot.png")
    print("- results/plots/prior_parameter_summary_table.png")
    print("- results/data/publication_ready_summary_table.csv")
    print("="*60)

if __name__ == "__main__":
    main()
