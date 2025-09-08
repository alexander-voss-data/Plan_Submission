"""
Analysis and statistical utilities for Trial Activation Analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_temporal_features(df):
    """
    Create temporal features for each row in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with temporal features added
    """
    logger.info("Creating row-level temporal features")
    
    df_features = df.copy()
    
    print("Creating temporal features:")
    
    # Trial duration in days
    df_features['trial_duration_days'] = (df_features['TRIAL_END'] - df_features['TRIAL_START']).dt.days
    print("  [CREATED] trial_duration_days")
    
    # Days since trial start
    df_features['days_since_trial_start'] = (df_features['TIMESTAMP'] - df_features['TRIAL_START']).dt.days
    print("  [CREATED] days_since_trial_start")
    
    # Days until trial end
    df_features['days_until_trial_end'] = (df_features['TRIAL_END'] - df_features['TIMESTAMP']).dt.days
    print("  [CREATED] days_until_trial_end")
    
    # Hour of day
    df_features['hour'] = df_features['TIMESTAMP'].dt.hour
    print("  [CREATED] hour")
    
    # Day of week (0=Monday, 6=Sunday)
    df_features['day_of_week'] = df_features['TIMESTAMP'].dt.dayofweek
    print("  [CREATED] day_of_week")
    
    # Is weekend flag
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    print("  [CREATED] is_weekend")
    
    # Trial week (1-4 based on days since trial start)
    df_features['trial_week'] = ((df_features['days_since_trial_start'] // 7) + 1).clip(upper=4)
    print("  [CREATED] trial_week")
    
    # Display feature statistics
    print(f"\nTemporal Feature Statistics:")
    temporal_features = ['trial_duration_days', 'days_since_trial_start', 'days_until_trial_end', 'hour', 'day_of_week', 'trial_week']
    
    for feature in temporal_features:
        values = df_features[feature]
        print(f"  {feature}:")
        print(f"    - Min: {values.min()}, Max: {values.max()}")
        print(f"    - Mean: {values.mean():.2f}, Std: {values.std():.2f}")
    
    logger.info("Row-level temporal features created successfully")
    return df_features

def create_activity_categories(df, utils):
    """
    Create activity categories based on the activity mapping.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with ACTIVITY_NAME column
    utils : dict
        Dictionary of utility functions
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with activity_category column added
    """
    logger.info("Creating activity categories")
    
    df_categorized = df.copy()
    
    # Define activity category mapping
    activity_category_mapping = {
        # Scheduling & Shifts
        'Scheduling.Availability.Set': 'Scheduling_Shifts',
        'Scheduling.Shift.Created': 'Scheduling_Shifts',
        'Scheduling.Shift.AssignmentChanged': 'Scheduling_Shifts',
        'Scheduling.Template.ApplyModal.Applied': 'Scheduling_Shifts',
        'Scheduling.ShiftSwap.Created': 'Scheduling_Shifts',
        'Scheduling.ShiftSwap.Accepted': 'Scheduling_Shifts',
        'Scheduling.ShiftHandover.Created': 'Scheduling_Shifts',
        'Scheduling.ShiftHandover.Accepted': 'Scheduling_Shifts',
        'Scheduling.OpenShiftRequest.Created': 'Scheduling_Shifts',
        'Scheduling.OpenShiftRequest.Approved': 'Scheduling_Shifts',
        'Mobile.Schedule.Loaded': 'Scheduling_Shifts',
        'Shift.View.Opened': 'Scheduling_Shifts',
        'ShiftDetails.View.Opened': 'Scheduling_Shifts',
        
        # Absence Management
        'Absence.Request.Created': 'Absence_Management',
        'Absence.Request.Approved': 'Absence_Management',
        'Absence.Request.Rejected': 'Absence_Management',
        
        # Time Tracking (Punch Clock)
        'PunchClock.PunchedIn': 'Time_Tracking',
        'Break.Activate.Started': 'Time_Tracking',
        'Break.Activate.Finished': 'Time_Tracking',
        'PunchClock.PunchedOut': 'Time_Tracking',
        'PunchClockStartNote.Add.Completed': 'Time_Tracking',
        'PunchClockEndNote.Add.Completed': 'Time_Tracking',
        'PunchClock.Entry.Edited': 'Time_Tracking',
        
        # Approvals & Payroll
        'Scheduling.Shift.Approved': 'Approvals_Payroll',
        'Timesheets.BulkApprove.Confirmed': 'Approvals_Payroll',
        'Integration.Xero.PayrollExport.Synced': 'Approvals_Payroll',
        
        # Financials
        'Revenue.Budgets.Created': 'Financials',
        
        # Communication
        'Communication.Message.Created': 'Communication'
    }
    
    print("Creating activity categories:")
    
    # Apply category mapping
    df_categorized['activity_category'] = df_categorized['ACTIVITY_NAME'].map(activity_category_mapping)
    
    # Check for unmapped activities
    unmapped = df_categorized[df_categorized['activity_category'].isna()]['ACTIVITY_NAME'].unique()
    if len(unmapped) > 0:
        print(f"  [WARNING] Found {len(unmapped)} unmapped activities:")
        for activity in unmapped:
            print(f"    - {activity}")
        df_categorized['activity_category'] = df_categorized['activity_category'].fillna('Other')
    else:
        print("  [COMPLETED] All activities successfully mapped to categories")
    
    # Display category distribution
    print(f"\nActivity Category Distribution:")
    category_counts = df_categorized['activity_category'].value_counts()
    total_activities = len(df_categorized)
    
    for category, count in category_counts.items():
        percentage = count / total_activities * 100
        print(f"  {category}: {utils['format_number'](count)} ({utils['format_percentage'](percentage, 1)})")
    
    logger.info("Activity categories created successfully")
    
    return df_categorized

def create_organization_level_features(df):
    """
    Create aggregated features at the organization level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with temporal and category features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with organization-level features
    """
    logger.info("Creating organization-level aggregated features")
    
    print("Creating organization-level aggregated features:")
    
    org_features = []
    
    for org_id, org_data in df.groupby('ORGANIZATION_ID'):
        # Basic counts
        unique_activities_count = org_data['ACTIVITY_NAME'].nunique()
        total_activities_count = len(org_data)
        
        # Trial duration
        trial_duration = org_data['trial_duration_days'].iloc[0]
        
        # Activities per day
        activities_per_day = total_activities_count / trial_duration if trial_duration > 0 else 0
        
        # Category features
        unique_categories_count = org_data['activity_category'].nunique()
        
        # Temporal features
        days_active = org_data['days_since_trial_start'].nunique()
        
        # Activity density
        activity_density = activities_per_day / days_active if days_active > 0 else 0
        
        # Time to first activity (hours from trial start)
        first_activity_time = org_data['TIMESTAMP'].min()
        trial_start_time = org_data['TRIAL_START'].iloc[0]
        time_to_first_activity = (first_activity_time - trial_start_time).total_seconds() / 3600
        
        # Activity spread
        activity_spread = org_data['days_since_trial_start'].std() if len(org_data) > 1 else 0
        
        # Weekly timing features
        week1_data = org_data[org_data['days_since_trial_start'] < 7]
        week1_activity_count = len(week1_data)
        week1_activity_ratio = week1_activity_count / total_activities_count if total_activities_count > 0 else 0
        
        # Early vs late activity ratio
        early_data = org_data[org_data['days_since_trial_start'] < 7]
        late_data = org_data[org_data['days_since_trial_start'] >= 7]
        early_vs_late_ratio = len(early_data) / len(late_data) if len(late_data) > 0 else float('inf')
        
        # Front-loaded pattern
        front_loaded_pattern = 1 if week1_activity_ratio > 0.5 else 0
        
        # Week 1 engagement binary
        week1_engagement_binary = 1 if week1_activity_count > 0 else 0
        
        # Weekly activity distribution
        activities_week1 = len(org_data[org_data['days_since_trial_start'] < 7])
        activities_week2 = len(org_data[(org_data['days_since_trial_start'] >= 7) & (org_data['days_since_trial_start'] < 14)])
        activities_week3 = len(org_data[(org_data['days_since_trial_start'] >= 14) & (org_data['days_since_trial_start'] < 21)])
        activities_week4_plus = len(org_data[org_data['days_since_trial_start'] >= 21])
        
        # Conversion info
        converted = org_data['CONVERTED'].iloc[0]
        
        org_features.append({
            'ORGANIZATION_ID': org_id,
            'unique_activities_count': unique_activities_count,
            'total_activities_count': total_activities_count,
            'activities_per_day': activities_per_day,
            'unique_categories_count': unique_categories_count,
            'days_active': days_active,
            'activity_density': activity_density,
            'time_to_first_activity': time_to_first_activity,
            'activity_spread': activity_spread,
            'week1_activity_count': week1_activity_count,
            'week1_activity_ratio': week1_activity_ratio,
            'early_vs_late_ratio': early_vs_late_ratio,
            'front_loaded_pattern': front_loaded_pattern,
            'week1_engagement_binary': week1_engagement_binary,
            'activities_week1': activities_week1,
            'activities_week2': activities_week2,
            'activities_week3': activities_week3,
            'activities_week4_plus': activities_week4_plus,
            'converted': converted
        })
    
    # Convert to DataFrame
    org_features_df = pd.DataFrame(org_features)
    
    logger.info("Organization-level features created successfully")
    logger.info(f"Created features for {len(org_features_df)} organizations")
    
    return org_features_df

def create_dataset_segments(df):
    """
    Create three analysis segments for comparative analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all features
        
    Returns:
    --------
    tuple
        (segment_1, segment_2, segment_3) dataframes
    """
    logger.info("Creating dataset segments for analysis")
    
    print("Creating analysis segments:")
    print("  Focus: Compare segments 1 and 2 for conversion indicators")
    
    # Segment 1: Non-converted organizations
    segment_not_converted = df[df['CONVERTED'] == False].copy()
    print(f"  [SEGMENT 1] Segment Not Converted: {len(segment_not_converted):,} records")
    print(f"    - Organizations: {segment_not_converted['ORGANIZATION_ID'].nunique()}")
    
    # Segment 2: Pre-conversion activities only
    converted_df = df[df['CONVERTED'] == True].copy()
    
    # Convert CONVERTED_AT to datetime for comparison
    converted_df['CONVERTED_AT_dt'] = pd.to_datetime(converted_df['CONVERTED_AT'])
    
    # Keep only activities that occurred before conversion
    segment_pre_conversion = converted_df[converted_df['TIMESTAMP'] < converted_df['CONVERTED_AT_dt']].copy()
    print(f"  [SEGMENT 2] Segment Pre-Conversion: {len(segment_pre_conversion):,} records")
    print(f"    - Organizations: {segment_pre_conversion['ORGANIZATION_ID'].nunique()}")
    
    # Segment 3: All converted organization activities
    segment_all_converted = converted_df.copy()
    print(f"  [SEGMENT 3] Segment All Converted: {len(segment_all_converted):,} records")
    print(f"    - Organizations: {segment_all_converted['ORGANIZATION_ID'].nunique()}")
    
    logger.info(f"Dataset segmentation completed")
    
    return segment_not_converted, segment_pre_conversion, segment_all_converted

def perform_univariate_analysis(segment_not_converted, segment_pre_conversion, segment_all_converted):
    """
    Perform univariate analysis comparing segments.
    
    Parameters:
    -----------
    segment_not_converted : pd.DataFrame
        Non-converted organizations
    segment_pre_conversion : pd.DataFrame  
        Pre-conversion activities
    segment_all_converted : pd.DataFrame
        All converted activities
    """
    logger.info("Starting univariate analysis")
    
    print("Univariate Analysis:")
    print("-" * 50)
    
    # Conversion rate analysis
    total_orgs = segment_not_converted['ORGANIZATION_ID'].nunique() + segment_pre_conversion['ORGANIZATION_ID'].nunique()
    converted_orgs = segment_pre_conversion['ORGANIZATION_ID'].nunique()
    conversion_rate = converted_orgs / total_orgs * 100
    
    print(f"Conversion Rate Analysis:")
    print(f"  - Total organizations: {total_orgs:,}")
    print(f"  - Converted organizations: {converted_orgs:,}")
    print(f"  - Overall conversion rate: {conversion_rate:.2f}%")
    
    # Activity distribution comparison
    print(f"\nActivity Distribution (Not Converted vs Pre-Conversion):")
    
    activities_not_conv = segment_not_converted['ACTIVITY_NAME'].value_counts()
    activities_pre_conv = segment_pre_conversion['ACTIVITY_NAME'].value_counts()
    
    all_activities = set(activities_not_conv.index) | set(activities_pre_conv.index)
    
    print(f"  Top 10 activities comparison:")
    print(f"  {'Activity':<40} {'Not Converted':<15} {'Pre-Conversion':<15} {'Difference'}")
    print(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*10}")
    
    activity_comparison = []
    for activity in sorted(all_activities):
        not_conv_count = activities_not_conv.get(activity, 0)
        pre_conv_count = activities_pre_conv.get(activity, 0)
        diff = pre_conv_count - not_conv_count
        activity_comparison.append((activity, not_conv_count, pre_conv_count, diff))
    
    # Sort by absolute difference and show top 10
    activity_comparison.sort(key=lambda x: abs(x[3]), reverse=True)
    for activity, not_conv_count, pre_conv_count, diff in activity_comparison[:10]:
        activity_short = activity[:35] + "..." if len(activity) > 35 else activity
        print(f"  {activity_short:<40} {not_conv_count:<15} {pre_conv_count:<15} {diff:+}")
    
    logger.info("Univariate analysis completed")
    
    return {
        'conversion_rate': conversion_rate,
        'total_orgs': total_orgs,
        'converted_orgs': converted_orgs,
        'activity_comparison': activity_comparison[:20]  # Return top 20 for further analysis
    }

def perform_bivariate_analysis(segment_not_converted, segment_pre_conversion, org_features_df):
    """
    Perform bivariate analysis between segments and features.
    
    Parameters:
    -----------
    segment_not_converted : pd.DataFrame
        Non-converted segment
    segment_pre_conversion : pd.DataFrame
        Pre-conversion segment
    org_features_df : pd.DataFrame
        Organization-level features
    """
    logger.info("Starting bivariate analysis")
    
    print("Bivariate Analysis:")
    print("-" * 50)
    
    # Separate organization features by conversion status
    converted_orgs = org_features_df[org_features_df['converted'] == True]
    non_converted_orgs = org_features_df[org_features_df['converted'] == False]
    
    print(f"Organization-Level Feature Comparison:")
    
    # Key features for comparison
    key_features = [
        'total_activities_count', 'unique_activities_count', 'unique_categories_count',
        'activities_per_day', 'days_active', 'week1_activity_count', 
        'week1_activity_ratio', 'front_loaded_pattern'
    ]
    
    print(f"  {'Feature':<25} {'Converted':<12} {'Non-Conv':<12} {'Difference':<12} {'% Diff'}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    comparison_results = {}
    
    for feature in key_features:
        if feature in converted_orgs.columns:
            conv_mean = converted_orgs[feature].mean()
            non_conv_mean = non_converted_orgs[feature].mean()
            difference = conv_mean - non_conv_mean
            pct_diff = (difference / non_conv_mean * 100) if non_conv_mean != 0 else 0
            
            comparison_results[feature] = {
                'converted_mean': conv_mean,
                'non_converted_mean': non_conv_mean,
                'difference': difference,
                'percent_difference': pct_diff
            }
            
            print(f"  {feature:<25} {conv_mean:<12.2f} {non_conv_mean:<12.2f} {difference:<12.2f} {pct_diff:<8.1f}%")
    
    logger.info("Bivariate analysis completed")
    
    return comparison_results

def perform_advanced_eda(segment_not_converted, segment_pre_conversion, org_features_df):
    """
    Perform advanced exploratory data analysis and pattern discovery.
    
    Parameters:
    -----------
    segment_not_converted : pd.DataFrame
        Non-converted segment
    segment_pre_conversion : pd.DataFrame
        Pre-conversion segment  
    org_features_df : pd.DataFrame
        Organization features
    """
    logger.info("Starting advanced EDA")
    
    print("Advanced EDA and Pattern Discovery:")
    print("-" * 50)
    
    # Timing pattern analysis
    print(f"Weekly Timing Pattern Analysis:")
    
    converted_orgs = org_features_df[org_features_df['converted'] == True]
    non_converted_orgs = org_features_df[org_features_df['converted'] == False]
    
    # Week 1 engagement patterns
    conv_week1_engagement = converted_orgs['week1_engagement_binary'].mean() * 100
    non_conv_week1_engagement = non_converted_orgs['week1_engagement_binary'].mean() * 100
    
    print(f"  Week 1 engagement rate:")
    print(f"    Converted: {conv_week1_engagement:.1f}%")
    print(f"    Non-converted: {non_conv_week1_engagement:.1f}%")
    print(f"    Difference: {conv_week1_engagement - non_conv_week1_engagement:+.1f} percentage points")
    
    # Front-loaded pattern analysis
    conv_front_loaded = converted_orgs['front_loaded_pattern'].mean() * 100
    non_conv_front_loaded = non_converted_orgs['front_loaded_pattern'].mean() * 100
    
    print(f"  Front-loaded pattern (>50% activities in week 1):")
    print(f"    Converted: {conv_front_loaded:.1f}%")
    print(f"    Non-converted: {non_conv_front_loaded:.1f}%")
    print(f"    Difference: {conv_front_loaded - non_conv_front_loaded:+.1f} percentage points")
    
    # Activity breadth analysis
    print(f"\nActivity Breadth Analysis:")
    
    conv_unique_activities = converted_orgs['unique_activities_count'].mean()
    non_conv_unique_activities = non_converted_orgs['unique_activities_count'].mean()
    
    conv_categories = converted_orgs['unique_categories_count'].mean()
    non_conv_categories = non_converted_orgs['unique_categories_count'].mean()
    
    print(f"  Average unique activities:")
    print(f"    Converted: {conv_unique_activities:.1f}")
    print(f"    Non-converted: {non_conv_unique_activities:.1f}")
    print(f"    Difference: {conv_unique_activities - non_conv_unique_activities:+.1f}")
    
    print(f"  Average categories used:")
    print(f"    Converted: {conv_categories:.1f}")
    print(f"    Non-converted: {non_conv_categories:.1f}")
    print(f"    Difference: {conv_categories - non_conv_categories:+.1f}")
    
    logger.info("Advanced EDA completed")
    
    return {
        'week1_engagement_diff': conv_week1_engagement - non_conv_week1_engagement,
        'front_loaded_diff': conv_front_loaded - non_conv_front_loaded,
        'unique_activities_diff': conv_unique_activities - non_conv_unique_activities,
        'categories_diff': conv_categories - non_conv_categories
    }
