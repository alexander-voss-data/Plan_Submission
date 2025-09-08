"""
Data cleaning and validation utilities for Trial Activation Analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def identify_and_remove_duplicates(df, utils):
    """
    Identify and remove duplicates based on ORGANIZATION_ID + ACTIVITY_NAME + TIMESTAMP.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    utils : dict
        Dictionary of utility functions
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    logger.info("Starting duplicate identification and removal")
    
    # Store original shape
    original_shape = df.shape
    
    # Define duplicate columns
    duplicate_cols = ['ORGANIZATION_ID', 'ACTIVITY_NAME', 'TIMESTAMP']
    
    # Identify duplicates
    duplicates = df.duplicated(subset=duplicate_cols, keep='first')
    num_duplicates = duplicates.sum()
    
    # Log duplicate patterns before removal
    if num_duplicates > 0:
        print(f"\nDuplicate Analysis:")
        print(f"  - Total duplicates found: {utils['format_number'](num_duplicates)}")
        print(f"  - Percentage of total data: {utils['format_percentage'](num_duplicates/len(df)*100, 2)}")
        
        # Analyze duplicate patterns
        duplicate_records = df[duplicates]
        print(f"  - Organizations with duplicates: {duplicate_records['ORGANIZATION_ID'].nunique()}")
        print(f"  - Activities with duplicates: {duplicate_records['ACTIVITY_NAME'].nunique()}")
        
        # Show top activities with duplicates
        duplicate_activity_counts = duplicate_records['ACTIVITY_NAME'].value_counts().head(10)
        print(f"\n  Top activities with duplicates:")
        for activity, count in duplicate_activity_counts.items():
            print(f"    - {activity}: {count}")
    
    # Remove duplicates
    df_clean = df.drop_duplicates(subset=duplicate_cols, keep='first')
    final_shape = df_clean.shape
    
    # Log impact
    rows_removed = original_shape[0] - final_shape[0]
    logger.info(f"Duplicates removed: {rows_removed}")
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Final shape: {final_shape}")
    
    print(f"\n[COMPLETED] Duplicate removal completed:")
    print(f"  - Original rows: {utils['format_number'](original_shape[0])}")
    print(f"  - Duplicates removed: {utils['format_number'](rows_removed)}")
    print(f"  - Final rows: {utils['format_number'](final_shape[0])}")
    print(f"  - Data reduction: {utils['format_percentage'](rows_removed/original_shape[0]*100, 2)}")
    
    return df_clean

def convert_and_validate_data_types(df):
    """
    Convert object columns to appropriate data types and validate conversions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple
        (DataFrame with proper data types, dict of conversion errors)
    """
    logger.info("Starting data type conversion and validation")
    
    df_converted = df.copy()
    conversion_errors = {}
    
    # Convert timestamp columns
    timestamp_columns = ['TIMESTAMP', 'CONVERTED_AT', 'TRIAL_START', 'TRIAL_END']
    
    for col in timestamp_columns:
        try:
            print(f"Converting {col} to datetime...")
            original_nulls = df_converted[col].isnull().sum()
            
            # Handle CONVERTED_AT separately as it has legitimate nulls
            if col == 'CONVERTED_AT':
                # Convert non-null values only
                mask = df_converted[col].notna()
                df_converted.loc[mask, col] = pd.to_datetime(df_converted.loc[mask, col], errors='coerce')
            else:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
            
            # Check for conversion issues
            new_nulls = df_converted[col].isnull().sum()
            conversion_failures = new_nulls - original_nulls
            
            if conversion_failures > 0:
                conversion_errors[col] = conversion_failures
                logger.warning(f"  [WARNING] {conversion_failures} values failed conversion in {col}")
            else:
                print(f"  [SUCCESS] {col} converted successfully")
                
        except Exception as e:
            conversion_errors[col] = str(e)
            logger.error(f"  [ERROR] Error converting {col}: {str(e)}")
    
    # Validate timezone consistency
    print(f"\nTimezone Validation:")
    for col in timestamp_columns:
        if col in df_converted.columns and df_converted[col].dtype == 'datetime64[ns]':
            non_null_timestamps = df_converted[col].dropna()
            if len(non_null_timestamps) > 0:
                print(f"  {col}: {non_null_timestamps.min()} to {non_null_timestamps.max()}")
    
    # Ensure CONVERTED column is boolean
    print(f"\nBoolean Validation:")
    converted_col_type = df_converted['CONVERTED'].dtype
    print(f"  CONVERTED column type: {converted_col_type}")
    if converted_col_type != 'bool':
        df_converted['CONVERTED'] = df_converted['CONVERTED'].astype('bool')
        print(f"  [SUCCESS] CONVERTED column converted to boolean")
    else:
        print(f"  [SUCCESS] CONVERTED column already boolean")
    
    # Summary of conversions
    print(f"\nData Type Summary After Conversion:")
    for col, dtype in df_converted.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Log conversion errors if any
    if conversion_errors:
        logger.warning(f"Conversion errors encountered: {conversion_errors}")
    else:
        logger.info("All data type conversions completed successfully")
    
    return df_converted, conversion_errors

def analyze_missing_values(df, utils):
    """
    Analyze missing value patterns in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    utils : dict
        Dictionary of utility functions
        
    Returns:
    --------
    dict
        Summary of missing value analysis
    """
    logger.info("Starting missing value analysis")
    
    missing_summary = {}
    
    print("Missing Value Analysis:")
    print("-" * 50)
    
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / total_rows * 100
        
        missing_summary[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
        
        if missing_count > 0:
            print(f"  {col}: {utils['format_number'](missing_count)} ({utils['format_percentage'](missing_pct, 2)})")
        else:
            print(f"  {col}: No missing values [OK]")
    
    # Analyze CONVERTED_AT pattern specifically
    print(f"\nCONVERTED_AT Pattern Analysis:")
    converted_true_count = df[df['CONVERTED'] == True].shape[0]
    converted_false_count = df[df['CONVERTED'] == False].shape[0]
    converted_at_null_count = df['CONVERTED_AT'].isnull().sum()
    
    print(f"  - Organizations with CONVERTED = True: {utils['format_number'](converted_true_count)}")
    print(f"  - Organizations with CONVERTED = False: {utils['format_number'](converted_false_count)}")
    print(f"  - Records with CONVERTED_AT = NULL: {utils['format_number'](converted_at_null_count)}")
    
    # Verify business logic
    invalid_pattern = df[(df['CONVERTED'] == False) & (df['CONVERTED_AT'].notna())].shape[0]
    if invalid_pattern == 0:
        print(f"  [VALIDATED] Business logic validated: All CONVERTED=False records have NULL CONVERTED_AT")
    else:
        print(f"  [WARNING] Found {invalid_pattern} records where CONVERTED=False but CONVERTED_AT is not NULL")
    
    # Check if all converted organizations have CONVERTED_AT
    missing_converted_at = df[(df['CONVERTED'] == True) & (df['CONVERTED_AT'].isna())].shape[0]
    if missing_converted_at == 0:
        print(f"  [VALIDATED] All converted organizations have CONVERTED_AT timestamp")
    else:
        print(f"  [WARNING] Found {missing_converted_at} converted organizations missing CONVERTED_AT")
    
    logger.info("Missing value analysis completed")
    
    return missing_summary

def document_business_logic(df, utils):
    """Document the business logic for missing values."""
    print(f"\nBusiness Logic Documentation:")
    print("-" * 50)
    print("  Missing Value Rules:")
    print("  1. CONVERTED_AT is NULL for non-converted organizations (CONVERTED = False)")
    print("  2. CONVERTED_AT contains timestamp for converted organizations (CONVERTED = True)")
    print("  3. No imputation needed - missing values are meaningful business indicators")
    
    # Summary stats
    total_orgs = df['ORGANIZATION_ID'].nunique()
    converted_orgs = df[df['CONVERTED'] == True]['ORGANIZATION_ID'].nunique()
    conversion_rate = converted_orgs / total_orgs * 100
    
    print(f"\n  Key Metrics:")
    print(f"  - Total organizations: {utils['format_number'](total_orgs)}")
    print(f"  - Converted organizations: {utils['format_number'](converted_orgs)}")
    print(f"  - Overall conversion rate: {utils['format_percentage'](conversion_rate, 2)}")

def validate_data_rules(df, utils):
    """
    Validate business rules and data integrity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    utils : dict
        Dictionary of utility functions
        
    Returns:
    --------
    tuple
        (dict of validation results, list of anomalous record indices)
    """
    logger.info("Starting data validation rules check")
    
    validation_results = {}
    anomalies = []
    
    print("Data Validation Rules:")
    print("-" * 60)
    
    # Rule 1: TIMESTAMP should fall between TRIAL_START and TRIAL_END
    print("  Rule 1: TIMESTAMP within trial period")
    invalid_timestamps = df[
        (df['TIMESTAMP'] < df['TRIAL_START']) | 
        (df['TIMESTAMP'] > df['TRIAL_END'])
    ]
    
    if len(invalid_timestamps) == 0:
        print("    [VALIDATED] All timestamps within trial period")
        validation_results['timestamp_within_trial'] = True
    else:
        print(f"    [WARNING] Found {len(invalid_timestamps)} records with timestamps outside trial period")
        validation_results['timestamp_within_trial'] = False
        anomalies.extend(invalid_timestamps.index.tolist())
    
    # Rule 2: CONVERTED_AT should be <= TRIAL_END when not null
    print("\n  Rule 2: CONVERTED_AT within trial period")
    converted_records = df[df['CONVERTED_AT'].notna()]
    
    if len(converted_records) > 0:
        converted_records_copy = converted_records.copy()
        converted_records_copy['CONVERTED_AT'] = pd.to_datetime(converted_records_copy['CONVERTED_AT'])
        
        invalid_converted_at = converted_records_copy[
            converted_records_copy['CONVERTED_AT'] > converted_records_copy['TRIAL_END']
        ]
        
        if len(invalid_converted_at) == 0:
            print("    [VALIDATED] All CONVERTED_AT timestamps within trial period")
            validation_results['converted_at_within_trial'] = True
        else:
            print(f"    [WARNING] Found {len(invalid_converted_at)} records with CONVERTED_AT after trial end")
            validation_results['converted_at_within_trial'] = False
    else:
        print("    [VALIDATED] No CONVERTED_AT timestamps to validate")
        validation_results['converted_at_within_trial'] = True
    
    # Rule 3: Validate trial duration
    print("\n  Rule 3: Trial duration validation")
    df_copy = df.copy()
    df_copy['trial_duration'] = (df_copy['TRIAL_END'] - df_copy['TRIAL_START']).dt.days
    
    org_trial_durations = df_copy.groupby('ORGANIZATION_ID')['trial_duration'].first()
    
    print(f"    Trial duration statistics:")
    print(f"      - Min duration: {org_trial_durations.min()} days")
    print(f"      - Max duration: {org_trial_durations.max()} days")
    print(f"      - Mean duration: {org_trial_durations.mean():.1f} days")
    
    expected_duration = 30
    duration_tolerance = 1
    
    invalid_durations = org_trial_durations[
        abs(org_trial_durations - expected_duration) > duration_tolerance
    ]
    
    if len(invalid_durations) == 0:
        print(f"    [VALIDATED] All trial durations within {expected_duration}±{duration_tolerance} days")
        validation_results['trial_duration_valid'] = True
    else:
        print(f"    [WARNING] Found {len(invalid_durations)} organizations with unusual trial duration")
        validation_results['trial_duration_valid'] = False
    
    # Summary
    print(f"\nValidation Summary:")
    print("-" * 40)
    total_rules = len(validation_results)
    passed_rules = sum(validation_results.values())
    
    print(f"  - Rules checked: {total_rules}")
    print(f"  - Rules passed: {passed_rules}")
    print(f"  - Success rate: {utils['format_percentage'](passed_rules/total_rules*100, 1)}")
    
    if len(anomalies) > 0:
        print(f"  - Total anomalous records: {len(set(anomalies))}")
    
    logger.info(f"Data validation completed. Results: {validation_results}")
    
    return validation_results, anomalies

def apply_statistical_filtering(df):
    """
    Apply filtering to remove organizations with too few or too many activities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    logger.info("Applying statistical robustness filtering")
    
    # Calculate organization-level activity counts
    org_activity_counts = df.groupby('ORGANIZATION_ID').size().reset_index(name='total_activities')
    
    # Define filtering criteria
    min_activities = 3
    max_activities_percentile = 0.95
    max_activities = org_activity_counts['total_activities'].quantile(max_activities_percentile)
    
    # Identify organizations to keep
    orgs_to_keep = org_activity_counts[
        (org_activity_counts['total_activities'] >= min_activities) &
        (org_activity_counts['total_activities'] <= max_activities)
    ]['ORGANIZATION_ID']
    
    # Apply filtering
    df_filtered = df[df['ORGANIZATION_ID'].isin(orgs_to_keep)].copy()
    
    # Calculate impact metrics
    original_records = len(df)
    original_orgs = df['ORGANIZATION_ID'].nunique()
    original_conv_rate = df['CONVERTED'].mean()
    
    filtered_records = len(df_filtered)
    filtered_orgs = df_filtered['ORGANIZATION_ID'].nunique()
    filtered_conv_rate = df_filtered['CONVERTED'].mean()
    
    records_removed = original_records - filtered_records
    orgs_removed = original_orgs - filtered_orgs
    
    print(f"\nStatistical Filtering Results:")
    print(f"  Filtering criteria:")
    print(f"    - Minimum activities per org: {min_activities}")
    print(f"    - Maximum activities per org: {max_activities:.0f} (95th percentile)")
    print(f"")
    print(f"  Impact on dataset:")
    print(f"    - Records: {original_records:,} -> {filtered_records:,} (-{records_removed:,}, {records_removed/original_records*100:.1f}%)")
    print(f"    - Organizations: {original_orgs:,} -> {filtered_orgs:,} (-{orgs_removed:,}, {orgs_removed/original_orgs*100:.1f}%)")
    print(f"    - Conversion rate: {original_conv_rate*100:.2f}% -> {filtered_conv_rate*100:.2f}%")
    
    logger.info(f"Statistical filtering completed: {original_records} -> {filtered_records} records")
    
    return df_filtered
