"""
Visualization utilities for Trial Activation Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_plot_directories():
    """Create necessary directories for saving plots."""
    directories = [
        os.path.join("Task1", "bivariate_analysis_plots"),
        os.path.join("Task1", "improved_activity_plots")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")
    
    return directories[0], directories[1]

def save_plot(fig, filename, plot_dir, dpi=300):
    """Save a plot to the specified directory."""
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    logger.info(f"Plot saved: {filepath}")
    return filepath

def create_activity_distribution_plot(df, activity_column='ACTIVITY_NAME', title="Activity Distribution"):
    """
    Create a distribution plot for activities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    activity_column : str
        Column name for activities
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Count activities
    activity_counts = df[activity_column].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create bar plot
    bars = ax.bar(range(len(activity_counts)), activity_counts.values)
    ax.set_xlabel('Activities')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Set x-axis labels
    ax.set_xticks(range(len(activity_counts)))
    ax.set_xticklabels(activity_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, activity_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(activity_counts.values)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def create_category_distribution_plot(df, category_column='activity_category', title="Category Distribution"):
    """
    Create a distribution plot for activity categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    category_column : str
        Column name for categories
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Count categories
    category_counts = df[category_column].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    bars = ax.bar(category_counts.index, category_counts.values)
    ax.set_xlabel('Activity Categories')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, category_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts.values)*0.01,
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_temporal_analysis_plot(df, time_column='TIMESTAMP', title="Temporal Activity Analysis"):
    """
    Create temporal analysis plots showing activity patterns over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_column : str
        Column name for timestamps
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # Hourly distribution
    df['hour'] = pd.to_datetime(df[time_column]).dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    axes[0, 0].bar(hourly_counts.index, hourly_counts.values)
    axes[0, 0].set_title('Activity Distribution by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Count')
    
    # Daily distribution
    df['day_of_week'] = pd.to_datetime(df[time_column]).dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df['day_of_week'].value_counts().reindex(day_order)
    axes[0, 1].bar(daily_counts.index, daily_counts.values)
    axes[0, 1].set_title('Activity Distribution by Day of Week')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Daily time series
    df['date'] = pd.to_datetime(df[time_column]).dt.date
    daily_series = df['date'].value_counts().sort_index()
    axes[1, 0].plot(daily_series.index, daily_series.values)
    axes[1, 0].set_title('Activity Count Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Daily Activity Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Weekend vs weekday
    df['is_weekend'] = pd.to_datetime(df[time_column]).dt.dayofweek.isin([5, 6])
    weekend_counts = df['is_weekend'].value_counts()
    weekend_labels = ['Weekday', 'Weekend']
    axes[1, 1].pie(weekend_counts.values, labels=weekend_labels, autopct='%1.1f%%')
    axes[1, 1].set_title('Weekend vs Weekday Activity')
    
    plt.tight_layout()
    return fig

def create_conversion_comparison_plot(converted_df, non_converted_df, feature_columns, title="Conversion Comparison"):
    """
    Create comparison plots between converted and non-converted organizations.
    
    Parameters:
    -----------
    converted_df : pd.DataFrame
        Converted organizations data
    non_converted_df : pd.DataFrame
        Non-converted organizations data
    feature_columns : list
        List of feature columns to compare
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle(title, fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(feature_columns):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create comparison histogram
        converted_values = converted_df[feature].dropna()
        non_converted_values = non_converted_df[feature].dropna()
        
        ax.hist(non_converted_values, alpha=0.7, label='Non-converted', bins=20, density=True)
        ax.hist(converted_values, alpha=0.7, label='Converted', bins=20, density=True)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, feature_columns, title="Feature Correlation Heatmap"):
    """
    Create a correlation heatmap for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list
        List of feature columns to include in correlation
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate correlation matrix
    correlation_matrix = df[feature_columns].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax, fmt='.2f')
    
    ax.set_title(title)
    plt.tight_layout()
    return fig

def create_activity_sequence_plot(df, org_ids=None, max_orgs=5, title="Activity Sequence Analysis"):
    """
    Create activity sequence visualization for selected organizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with TIMESTAMP, ORGANIZATION_ID, ACTIVITY_NAME
    org_ids : list
        List of organization IDs to plot (if None, select random sample)
    max_orgs : int
        Maximum number of organizations to plot
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if org_ids is None:
        # Select random sample of organizations
        unique_orgs = df['ORGANIZATION_ID'].unique()
        org_ids = np.random.choice(unique_orgs, min(max_orgs, len(unique_orgs)), replace=False)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(org_ids)))
    
    for i, org_id in enumerate(org_ids):
        org_data = df[df['ORGANIZATION_ID'] == org_id].sort_values('TIMESTAMP')
        
        # Convert timestamps to relative days
        start_time = org_data['TIMESTAMP'].min()
        org_data['days_from_start'] = (org_data['TIMESTAMP'] - start_time).dt.total_seconds() / (24 * 3600)
        
        # Plot activities
        y_pos = i * 1.0
        for j, (_, row) in enumerate(org_data.iterrows()):
            ax.scatter(row['days_from_start'], y_pos, c=[colors[i]], s=50, alpha=0.7)
            if j % 5 == 0:  # Label every 5th activity to avoid clutter
                ax.annotate(row['ACTIVITY_NAME'][:20], 
                          (row['days_from_start'], y_pos),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Days from First Activity')
    ax.set_ylabel('Organization')
    ax.set_title(title)
    ax.set_yticks(range(len(org_ids)))
    ax.set_yticklabels([f'Org {org_id}' for org_id in org_ids])
    
    plt.tight_layout()
    return fig
