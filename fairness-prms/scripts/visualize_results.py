#!/usr/bin/env python3
"""
Visualization script for fairness evaluation results.

Creates comprehensive charts and graphs to analyze:
- Accuracy trends across temperature settings and categories
- Fairness metrics (EOpp and EOdds gaps) distributions
- Performance vs fairness trade-offs
- Category-wise comparisons

Usage:
    python scripts/visualize_results.py

Output:
    Saves multiple chart images to evaluation_output/ directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set up professional seaborn styling with predefined color palettes
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2,
              rc={"axes.labelsize": 12, "axes.titlesize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11})

# Use predefined seaborn color palettes
category_palette = sns.color_palette("Set1", 11)  # For 11 categories
temp_palette = sns.color_palette("tab10", 4)  # For 4 temperature settings
metric_palette = sns.color_palette("Set2", 3)  # For accuracy, eopp, eodds

def load_evaluation_data(data_path):
    """Load evaluation results from pickle file."""
    return pd.read_pickle(data_path)

def create_accuracy_trends(df, output_dir):
    """Create accuracy trends across temperature settings."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Accuracy Trends Across Temperature Settings', fontsize=18, fontweight='bold', y=0.98)

    # Temperature order for consistent plotting
    temp_order = ['temp_001', 'temp_02', 'temp_04', 'temp_08']

    # 1. Line plot: Accuracy by temperature for each category
    for i, category in enumerate(df['category'].unique()):
        category_data = df[df['category'] == category].copy()
        category_data['temp_numeric'] = category_data['temp_setting'].map({
            'temp_001': 0.01, 'temp_02': 0.2, 'temp_04': 0.4, 'temp_08': 0.8
        })
        category_data = category_data.sort_values('temp_numeric')
        sns.lineplot(data=category_data, x='temp_numeric', y='accuracy',
                    marker='o', linewidth=2, label=category.replace('_', ' '),
                    color=category_palette[i % len(category_palette)], ax=ax1)

    ax1.set_xlabel('Temperature Setting')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Temperature by Category')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0.01, 0.2, 0.4, 0.8])
    ax1.set_xticklabels(['0.01', '0.2', '0.4', '0.8'])

    # 2. Box plot: Accuracy distribution by temperature
    sns.boxplot(data=df, x='temp_setting', y='accuracy', order=temp_order, ax=ax2)
    ax2.set_xlabel('Temperature Setting')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Distribution by Temperature')
    ax2.grid(True, alpha=0.3)

    # 3. Bar plot: Average accuracy by category
    category_avg = df.groupby('category')['accuracy'].mean().sort_values(ascending=True)
    sns.barplot(x=category_avg.values, y=[cat.replace('_', ' ') for cat in category_avg.index],
               ax=ax3, orient='h')
    ax3.set_xlabel('Average Accuracy')
    ax3.set_title('Average Accuracy by Category')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(category_avg.values):
        ax3.text(v + 0.01, i, '.3f', va='center')

    # 4. Heatmap: Accuracy by category and temperature
    pivot_data = df.pivot(index='category', columns='temp_setting', values='accuracy')
    pivot_data = pivot_data[temp_order]  # Ensure correct column order
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4,
                cbar_kws={'label': 'Accuracy', 'shrink': 0.8},
                annot_kws={'size': 10}, square=True, linewidths=0.5)
    ax4.set_title('Accuracy Heatmap: Category vs Temperature', fontsize=14, pad=20)
    ax4.set_xlabel('Temperature Setting', fontsize=12)
    ax4.set_ylabel('Category', fontsize=12)
    ax4.tick_params(axis='x', rotation=0)
    ax4.tick_params(axis='y', rotation=0)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(output_dir, 'accuracy_trends.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_fairness_analysis(df, output_dir):
    """Create fairness metrics analysis charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Fairness Metrics Analysis', fontsize=18, fontweight='bold', y=0.98)

    # 1. EOpp Gap distribution by category
    eopp_data = df.pivot(index='category', columns='temp_setting', values='eopp_gap')
    eopp_data_melted = eopp_data.reset_index().melt(id_vars='category', var_name='temp_setting', value_name='eopp_gap')
    sns.barplot(data=eopp_data_melted, x='category', y='eopp_gap', hue='temp_setting',
               palette=temp_palette, ax=ax1)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('EOpp Gap', fontsize=12)
    ax1.set_title('Equalized Opportunity Gap by Category and Temperature', fontsize=14, pad=20)
    ax1.legend(title='Temperature', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. EOdds Gap distribution by category
    eodds_data = df.pivot(index='category', columns='temp_setting', values='eodds_gap')
    eodds_data_melted = eodds_data.reset_index().melt(id_vars='category', var_name='temp_setting', value_name='eodds_gap')
    sns.barplot(data=eodds_data_melted, x='category', y='eodds_gap', hue='temp_setting',
               palette=temp_palette, ax=ax2)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('EOdds Gap', fontsize=12)
    ax2.set_title('Equalized Odds Gap by Category and Temperature', fontsize=14, pad=20)
    ax2.legend(title='Temperature', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Fairness vs Accuracy scatter plot
    temp_numeric = df['temp_setting'].map({'temp_001': 0.01, 'temp_02': 0.2, 'temp_04': 0.4, 'temp_08': 0.8})
    sns.scatterplot(data=df, x='accuracy', y='eopp_gap', hue=temp_numeric,
                   palette='viridis', alpha=0.7, s=60, ax=ax3)
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('EOpp Gap')
    ax3.set_title('Fairness vs Accuracy Trade-off')
    ax3.grid(True, alpha=0.3)

    # Add colorbar
    norm = mcolors.Normalize(temp_numeric.min(), temp_numeric.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Temperature')
    cbar.set_ticks([0.01, 0.2, 0.4, 0.8])
    cbar.set_ticklabels(['0.01', '0.2', '0.4', '0.8'])

    # 4. Fairness gap summary statistics
    fairness_summary = df.groupby('category')[['eopp_gap', 'eodds_gap']].agg(['mean', 'std']).round(4)
    fairness_summary.columns = ['_'.join(col).strip() for col in fairness_summary.columns]

    # Create a summary bar chart using seaborn
    fairness_melted = fairness_summary[['eopp_gap_mean', 'eodds_gap_mean']].reset_index()
    fairness_melted = fairness_melted.melt(id_vars='category', var_name='metric', value_name='value')
    fairness_melted['metric'] = fairness_melted['metric'].map({'eopp_gap_mean': 'EOpp Gap', 'eodds_gap_mean': 'EOdds Gap'})

    sns.barplot(data=fairness_melted, x='category', y='value', hue='metric',
               palette=['skyblue', 'lightcoral'], ax=ax4)
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Average Fairness Gap')
    ax4.set_title('Average Fairness Gaps by Category')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Metric')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(output_dir, 'fairness_analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_performance_overview(df, output_dir):
    """Create overall performance overview charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Performance Overview Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # 1. Overall accuracy distribution
    sns.histplot(data=df, x='accuracy', bins=20, ax=ax1, color=metric_palette[0], alpha=0.7, kde=True)
    ax1.axvline(df['accuracy'].mean(), color='red', linestyle='--', linewidth=2,
               label='.3f')
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Accuracy Distribution Across All Evaluations', fontsize=14, pad=20)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Fairness gap distributions
    sns.histplot(data=df, x='eopp_gap', bins=15, ax=ax2, color=metric_palette[1], alpha=0.7,
                label='EOpp Gap', kde=True)
    sns.histplot(data=df, x='eodds_gap', bins=15, ax=ax2, color=metric_palette[2], alpha=0.7,
                label='EOdds Gap', kde=True)
    ax2.axvline(df['eopp_gap'].mean(), color=sns.color_palette("Set1")[1], linestyle='--', linewidth=2,
               label='.3f')
    ax2.axvline(df['eodds_gap'].mean(), color=sns.color_palette("Set1")[2], linestyle='--', linewidth=2,
               label='.3f')
    ax2.set_xlabel('Fairness Gap', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Fairness Gap Distributions', fontsize=14, pad=20)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Temperature impact analysis
    temp_stats = df.groupby('temp_setting')[['accuracy', 'eopp_gap', 'eodds_gap']].mean()
    temp_stats = temp_stats.reindex(['temp_001', 'temp_02', 'temp_04', 'temp_08'])

    # Melt data for seaborn
    temp_melted = temp_stats.reset_index().melt(id_vars='temp_setting',
                                               var_name='metric', value_name='value')
    temp_melted['temp_numeric'] = temp_melted['temp_setting'].map({
        'temp_001': 0.01, 'temp_02': 0.2, 'temp_04': 0.4, 'temp_08': 0.8
    })

    sns.barplot(data=temp_melted, x='temp_setting', y='value', hue='metric',
               palette=metric_palette, ax=ax3,
               order=['temp_001', 'temp_02', 'temp_04', 'temp_08'])
    ax3.set_xlabel('Temperature Setting', fontsize=12)
    ax3.set_ylabel('Metric Value', fontsize=12)
    ax3.set_title('Average Metrics by Temperature Setting', fontsize=14, pad=20)
    ax3.legend(title='Metric', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Performance radar chart (simplified spider plot)
    # Calculate average performance by category
    radar_data = df.groupby('category')[['accuracy', 'eopp_gap', 'eodds_gap']].mean()

    # Normalize data for radar chart (0-1 scale)
    radar_normalized = radar_data.copy()
    radar_normalized['accuracy'] = (radar_normalized['accuracy'] - radar_normalized['accuracy'].min()) / \
                                  (radar_normalized['accuracy'].max() - radar_normalized['accuracy'].min())
    radar_normalized['eopp_gap'] = 1 - (radar_normalized['eopp_gap'] - radar_normalized['eopp_gap'].min()) / \
                                  (radar_normalized['eopp_gap'].max() - radar_normalized['eopp_gap'].min())
    radar_normalized['eodds_gap'] = 1 - (radar_normalized['eodds_gap'] - radar_normalized['eodds_gap'].min()) / \
                                   (radar_normalized['eodds_gap'].max() - radar_normalized['eodds_gap'].min())

    # Create radar chart for top 3 and bottom 3 categories by accuracy
    top_categories = radar_data.nlargest(3, 'accuracy').index.tolist()
    bottom_categories = radar_data.nsmallest(3, 'accuracy').index.tolist()

    # Plot radar for top performers
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for category in top_categories:
        values = radar_normalized.loc[category].values.tolist()
        values += values[:1]  # Close the loop
        ax4.plot(angles, values, 'o-', linewidth=2, label=category.replace('_', ' '))

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Accuracy\n(Normalized)', 'EOpp Fairness\n(Normalized)', 'EOdds Fairness\n(Normalized)'])
    ax4.set_title('Performance Radar: Top 3 Categories')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(output_dir, 'performance_overview.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_category_comparison(df, output_dir):
    """Create detailed category comparison charts."""
    categories = df['category'].unique()
    n_categories = len(categories)

    # Calculate number of rows needed (3 columns)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 7*n_rows))
    fig.suptitle('Category-wise Performance Comparison', fontsize=18, fontweight='bold', y=0.98)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, category in enumerate(sorted(categories)):
        row = i // n_cols
        col = i % n_cols

        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        category_data = df[df['category'] == category].copy()
        category_data['temp_numeric'] = category_data['temp_setting'].map({
            'temp_001': 0.01, 'temp_02': 0.2, 'temp_04': 0.4, 'temp_08': 0.8
        })
        category_data = category_data.sort_values('temp_numeric')

        # Melt data for seaborn plotting
        melted_data = category_data.melt(id_vars=['temp_numeric'],
                                       value_vars=['accuracy', 'eopp_gap', 'eodds_gap'],
                                       var_name='metric', value_name='value')

        # Create line plot with seaborn
        sns.lineplot(data=melted_data, x='temp_numeric', y='value', hue='metric',
                    style='metric', markers=['o', 's', '^'], ax=ax,
                    palette=metric_palette, linewidth=2.5, markersize=8)

        ax.set_xlabel('Temperature Setting', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{category.replace("_", " ")} Performance', fontsize=13, pad=15)
        ax.grid(True, alpha=0.3)

        # Set x-ticks
        ax.set_xticks([0.01, 0.2, 0.4, 0.8])
        ax.set_xticklabels(['0.01', '0.2', '0.4', '0.8'])

        # Position legend with better spacing
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)

    # Hide empty subplots
    for i in range(len(categories), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(output_dir, 'category_comparison.svg'), format='svg', bbox_inches='tight')
    plt.close()

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'evaluation_output' / 'evaluation_results.pkl'
    output_dir = project_root / 'evaluation_output'

    # Load data
    print("Loading evaluation results...")
    df = load_evaluation_data(data_path)
    print(f"Loaded {len(df)} evaluation records")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Generate all visualizations
    print("Creating accuracy trend visualizations...")
    create_accuracy_trends(df, output_dir)

    print("Creating fairness analysis visualizations...")
    create_fairness_analysis(df, output_dir)

    print("Creating performance overview dashboard...")
    create_performance_overview(df, output_dir)

    print("Creating detailed category comparisons...")
    create_category_comparison(df, output_dir)

    print("\nVisualization complete! Charts saved to:")
    print(f"  - {output_dir}/accuracy_trends.svg")
    print(f"  - {output_dir}/fairness_analysis.svg")
    print(f"  - {output_dir}/performance_overview.svg")
    print(f"  - {output_dir}/category_comparison.svg")

if __name__ == "__main__":
    main()