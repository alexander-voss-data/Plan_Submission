"""
Display and formatting utilities for Trial Activation Analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def print_phase_header(phase_number, phase_title):
    """Print a formatted phase header."""
    print(f"\n{'='*80}")
    print(f"PHASE {phase_number}: {phase_title.upper()}")
    print(f"{'='*80}")

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'-'*60}")
    print(f"Step {step_number}: {step_title}")
    print(f"{'-'*60}")

def create_utility_functions():
    """
    Create utility functions for consistent formatting throughout the analysis.
    
    Returns:
    --------
    dict
        Dictionary containing utility functions
    """
    logger.info("Creating utility functions")
    
    def format_number(num):
        """Format large numbers with commas."""
        if pd.isna(num):
            return "N/A"
        return f"{num:,.0f}" if isinstance(num, (int, float)) else str(num)
    
    def format_percentage(pct, decimals=1):
        """Format percentage with specified decimal places."""
        if pd.isna(pct):
            return "N/A%"
        return f"{pct:.{decimals}f}%"
    
    def format_currency(amount, symbol="$"):
        """Format currency values."""
        if pd.isna(amount):
            return f"{symbol}N/A"
        return f"{symbol}{amount:,.2f}"
    
    def format_duration(seconds):
        """Format duration in seconds to human readable format."""
        if pd.isna(seconds):
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def safe_divide(numerator, denominator, default=0):
        """Safely divide two numbers, returning default if denominator is 0."""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    def calculate_percentage_change(old_value, new_value):
        """Calculate percentage change between two values."""
        if pd.isna(old_value) or pd.isna(new_value) or old_value == 0:
            return None
        return ((new_value - old_value) / old_value) * 100
    
    def summarize_dataframe(df, title="DataFrame Summary"):
        """Print a comprehensive summary of a dataframe."""
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if df.shape[0] > 0:
            print(f"\nColumn Information:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                print(f"  {col:25} | {str(dtype):15} | "
                      f"Null: {null_count:6,} ({null_pct:5.1f}%) | "
                      f"Unique: {unique_count:8,}")
    
    def display_value_counts(series, title=None, top_n=10):
        """Display value counts for a series with formatting."""
        if title:
            print(f"\n{title}")
            print("-" * len(title))
        
        value_counts = series.value_counts().head(top_n)
        total = len(series)
        
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            print(f"  {str(value):30}: {format_number(count):>8} ({format_percentage(percentage)})")
        
        if len(series.unique()) > top_n:
            remaining = len(series.unique()) - top_n
            print(f"  ... and {remaining} more unique values")
    
    def compare_segments(segment1, segment2, metric_columns, segment1_name="Segment 1", segment2_name="Segment 2"):
        """Compare metrics between two data segments."""
        print(f"\nComparison: {segment1_name} vs {segment2_name}")
        print("-" * 60)
        
        for metric in metric_columns:
            if metric in segment1.columns and metric in segment2.columns:
                val1 = segment1[metric].mean()
                val2 = segment2[metric].mean()
                
                # Calculate difference and percentage change
                diff = val2 - val1
                pct_change = calculate_percentage_change(val1, val2)
                
                print(f"  {metric:30}:")
                print(f"    {segment1_name:15}: {val1:8.2f}")
                print(f"    {segment2_name:15}: {val2:8.2f}")
                print(f"    Difference:      {diff:8.2f}")
                if pct_change is not None:
                    print(f"    % Change:        {format_percentage(pct_change)}")
                print()
    
    utilities = {
        'format_number': format_number,
        'format_percentage': format_percentage,
        'format_currency': format_currency,
        'format_duration': format_duration,
        'safe_divide': safe_divide,
        'calculate_percentage_change': calculate_percentage_change,
        'summarize_dataframe': summarize_dataframe,
        'display_value_counts': display_value_counts,
        'compare_segments': compare_segments
    }
    
    logger.info("Utility functions created successfully")
    return utilities

def display_library_versions():
    """Display versions of key libraries for reproducibility."""
    print("\nLibrary Versions (for reproducibility):")
    print("-" * 50)
    
    libraries = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 
        'sklearn', 'statsmodels'
    ]
    
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  {lib:15}: {version}")
        except ImportError:
            print(f"  {lib:15}: Not installed")

def log_phase_completion(phase_number, phase_title, metrics=None):
    """Log the completion of a phase with optional metrics."""
    logger.info(f"Phase {phase_number} ({phase_title}) completed successfully")
    
    if metrics:
        logger.info(f"Phase {phase_number} metrics: {metrics}")
    
    print(f"\n[COMPLETED] Phase {phase_number}: {phase_title} - COMPLETED")
    
    if metrics:
        print("  Key metrics:")
        for key, value in metrics.items():
            print(f"    - {key}: {value}")

def create_summary_report(df, analysis_results=None):
    """Create a comprehensive summary report of the analysis."""
    print(f"\n{'='*80}")
    print("TRIAL ACTIVATION ANALYSIS - SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset overview
    print(f"\nDataset Overview:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Total organizations: {df['ORGANIZATION_ID'].nunique():,}")
    print(f"  - Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    print(f"  - Unique activities: {df['ACTIVITY_NAME'].nunique()}")
    
    # Conversion metrics
    if 'CONVERTED' in df.columns:
        total_orgs = df['ORGANIZATION_ID'].nunique()
        converted_orgs = df[df['CONVERTED'] == True]['ORGANIZATION_ID'].nunique()
        conversion_rate = (converted_orgs / total_orgs) * 100
        
        print(f"\nConversion Metrics:")
        print(f"  - Converted organizations: {converted_orgs:,}")
        print(f"  - Total organizations: {total_orgs:,}")
        print(f"  - Conversion rate: {conversion_rate:.2f}%")
    
    # Analysis results
    if analysis_results:
        print(f"\nAnalysis Results:")
        for key, value in analysis_results.items():
            print(f"  - {key}: {value}")
    
    print(f"\n{'='*80}")

def print_error_summary(errors):
    """Print a summary of errors encountered during analysis."""
    if not errors:
        print("[OK] No errors encountered during analysis")
        return
    
    print(f"\n[WARNING] Error Summary ({len(errors)} errors):")
    print("-" * 40)
    
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error['type']}: {error['message']}")
        if 'details' in error:
            print(f"     Details: {error['details']}")
    
    logger.warning(f"Analysis completed with {len(errors)} errors")
