"""
Statistical testing and modeling utilities for Trial Activation Analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, ks_2samp
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def perform_focused_conversion_hypothesis_tests(org_features_df, segment_not_converted, segment_pre_conversion):
    """
    Perform focused statistical tests on conversion differences.
    
    Parameters:
    -----------
    org_features_df : pd.DataFrame
        Organization-level features 
    segment_not_converted : pd.DataFrame
        Non-converted organization activities
    segment_pre_conversion : pd.DataFrame
        Pre-conversion activities
        
    Returns:
    --------
    dict
        Statistical test results on conversion drivers
    """
    logger.info("Starting focused conversion hypothesis testing")
    
    print("Focused Conversion Driver Hypothesis Testing:")
    print("=" * 60)
    
    # Create results storage
    statistical_results = {}
    
    # Separate converted and non-converted organizations
    converted_orgs = org_features_df[org_features_df['converted'] == True]
    non_converted_orgs = org_features_df[org_features_df['converted'] == False]
    
    print(f"Dataset Overview:")
    print(f"   Total organizations: {len(org_features_df):,}")
    print(f"   Converted: {len(converted_orgs):,} ({len(converted_orgs)/len(org_features_df)*100:.1f}%)")
    print(f"   Non-converted: {len(non_converted_orgs):,} ({len(non_converted_orgs)/len(org_features_df)*100:.1f}%)")
    
    # =============================================================================
    # 1. WEEKLY TIMING PATTERN TESTS
    # =============================================================================
    
    print(f"\n1. WEEKLY TIMING PATTERN TESTS:")
    print("-" * 40)
    
    # Test 1.1: Week 1 Engagement Binary (Chi-square)
    week1_engagement_converted = converted_orgs['week1_engagement_binary'].values
    week1_engagement_non_converted = non_converted_orgs['week1_engagement_binary'].values
    
    # Create contingency table using proper method
    group_labels = (['Converted']*len(week1_engagement_converted) + 
                   ['Non-converted']*len(week1_engagement_non_converted))
    engagement_values = (list(week1_engagement_converted) + 
                        list(week1_engagement_non_converted))
    
    # Create DataFrame to avoid index issues
    df_for_crosstab = pd.DataFrame({
        'group': group_labels,
        'week1_engaged': engagement_values
    })
    
    contingency_table = pd.crosstab(df_for_crosstab['group'], df_for_crosstab['week1_engaged'])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    statistical_results['week1_engagement_chi2'] = {
        'test': 'Chi-square test',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05
    }
    
    print(f"  Test 1.1 - Week 1 Engagement (Chi-square):")
    print(f"    Chi-square statistic: {chi2:.4f}")
    print(f"    P-value: {p_value:.6f}")
    print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
    
    # Test 1.2: Week 1 Activity Ratio (T-test)
    week1_ratio_converted = converted_orgs['week1_activity_ratio'].values
    week1_ratio_non_converted = non_converted_orgs['week1_activity_ratio'].values
    
    t_stat, t_p_value = ttest_ind(week1_ratio_converted, week1_ratio_non_converted)
    
    statistical_results['week1_ratio_ttest'] = {
        'test': 'Independent t-test',
        't_statistic': t_stat,
        'p_value': t_p_value,
        'converted_mean': week1_ratio_converted.mean(),
        'non_converted_mean': week1_ratio_non_converted.mean(),
        'significant': t_p_value < 0.05
    }
    
    print(f"  Test 1.2 - Week 1 Activity Ratio (T-test):")
    print(f"    T-statistic: {t_stat:.4f}")
    print(f"    P-value: {t_p_value:.6f}")
    print(f"    Converted mean: {week1_ratio_converted.mean():.4f}")
    print(f"    Non-converted mean: {week1_ratio_non_converted.mean():.4f}")
    print(f"    Significant: {'Yes' if t_p_value < 0.05 else 'No'} (α = 0.05)")
    
    # =============================================================================
    # 2. ACTIVITY BREADTH TESTS
    # =============================================================================
    
    print(f"\n2. ACTIVITY BREADTH TESTS:")
    print("-" * 40)
    
    # Test 2.1: Total Activities Count
    total_activities_converted = converted_orgs['total_activities_count'].values
    total_activities_non_converted = non_converted_orgs['total_activities_count'].values
    
    # Use Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = mannwhitneyu(total_activities_converted, total_activities_non_converted, alternative='two-sided')
    
    statistical_results['total_activities_mannwhitney'] = {
        'test': 'Mann-Whitney U test',
        'u_statistic': u_stat,
        'p_value': u_p_value,
        'converted_median': np.median(total_activities_converted),
        'non_converted_median': np.median(total_activities_non_converted),
        'significant': u_p_value < 0.05
    }
    
    print(f"  Test 2.1 - Total Activities Count (Mann-Whitney U):")
    print(f"    U-statistic: {u_stat:.4f}")
    print(f"    P-value: {u_p_value:.6f}")
    print(f"    Converted median: {np.median(total_activities_converted):.1f}")
    print(f"    Non-converted median: {np.median(total_activities_non_converted):.1f}")
    print(f"    Significant: {'Yes' if u_p_value < 0.05 else 'No'} (α = 0.05)")
    
    # Test 2.2: Unique Categories Count
    categories_converted = converted_orgs['unique_categories_count'].values
    categories_non_converted = non_converted_orgs['unique_categories_count'].values
    
    u_stat_cat, u_p_value_cat = mannwhitneyu(categories_converted, categories_non_converted, alternative='two-sided')
    
    statistical_results['categories_mannwhitney'] = {
        'test': 'Mann-Whitney U test',
        'u_statistic': u_stat_cat,
        'p_value': u_p_value_cat,
        'converted_median': np.median(categories_converted),
        'non_converted_median': np.median(categories_non_converted),
        'significant': u_p_value_cat < 0.05
    }
    
    print(f"  Test 2.2 - Unique Categories Count (Mann-Whitney U):")
    print(f"    U-statistic: {u_stat_cat:.4f}")
    print(f"    P-value: {u_p_value_cat:.6f}")
    print(f"    Converted median: {np.median(categories_converted):.1f}")
    print(f"    Non-converted median: {np.median(categories_non_converted):.1f}")
    print(f"    Significant: {'Yes' if u_p_value_cat < 0.05 else 'No'} (α = 0.05)")
    
    # =============================================================================
    # Summary of Statistical Tests
    # =============================================================================
    
    print(f"\nSTATISTICAL TESTS SUMMARY:")
    print("=" * 60)
    
    significant_tests = [test for test, results in statistical_results.items() if results.get('significant', False)]
    
    print(f"Total tests performed: {len(statistical_results)}")
    print(f"Significant tests (α = 0.05): {len(significant_tests)}")
    
    if significant_tests:
        print(f"\nSignificant findings:")
        for test in significant_tests:
            results = statistical_results[test]
            print(f"  [SIGNIFICANT] {test}: {results['test']} (p = {results['p_value']:.6f})")
    else:
        print(f"\nNo statistically significant differences found at α = 0.05 level.")
    
    logger.info(f"Statistical testing completed. {len(significant_tests)}/{len(statistical_results)} tests significant")
    
    return statistical_results

def build_predictive_models(org_features_df):
    """
    Build and evaluate predictive models for trial conversion.
    
    Parameters:
    -----------
    org_features_df : pd.DataFrame
        Organization-level features with conversion target
        
    Returns:
    --------
    dict
        Model results and performance metrics
    """
    logger.info("Building predictive models")
    
    print("Predictive Modeling:")
    print("=" * 50)
    
    # Prepare features and target
    feature_columns = [
        'total_activities_count', 'unique_activities_count', 'unique_categories_count',
        'activities_per_day', 'days_active', 'week1_activity_count', 
        'week1_activity_ratio', 'front_loaded_pattern', 'week1_engagement_binary'
    ]
    
    X = org_features_df[feature_columns].fillna(0)
    y = org_features_df['converted'].astype(int)
    
    print(f"Dataset for modeling:")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Samples: {len(X)}")
    print(f"  Positive class (converted): {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # =============================================================================
    # Model 1: Logistic Regression
    # =============================================================================
    
    print(f"\nModel 1: Logistic Regression")
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Test predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    test_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    
    models['logistic_regression'] = {
        'model': lr_model,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': test_auc_lr,
        'feature_importance': dict(zip(feature_columns, lr_model.coef_[0]))
    }
    
    print(f"  Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test AUC: {test_auc_lr:.4f}")
    
    # Feature importance
    print(f"  Top 3 features (by coefficient magnitude):")
    feature_importance = dict(zip(feature_columns, np.abs(lr_model.coef_[0])))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    for feature, importance in top_features:
        print(f"    {feature}: {importance:.4f}")
    
    # =============================================================================
    # Model 2: Random Forest
    # =============================================================================
    
    print(f"\nModel 2: Random Forest")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Test predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    test_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    
    models['random_forest'] = {
        'model': rf_model,
        'cv_auc_mean': cv_scores_rf.mean(),
        'cv_auc_std': cv_scores_rf.std(),
        'test_auc': test_auc_rf,
        'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
    }
    
    print(f"  Cross-validation AUC: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    print(f"  Test AUC: {test_auc_rf:.4f}")
    
    # Feature importance
    print(f"  Top 3 features (by importance):")
    rf_importance = dict(zip(feature_columns, rf_model.feature_importances_))
    top_features_rf = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    for feature, importance in top_features_rf:
        print(f"    {feature}: {importance:.4f}")
    
    # =============================================================================
    # Model Comparison
    # =============================================================================
    
    print(f"\nModel Comparison:")
    print(f"  {'Model':<20} {'CV AUC':<15} {'Test AUC':<15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'Logistic Regression':<20} {cv_scores.mean():.4f} ± {cv_scores.std():.3f}  {test_auc_lr:.4f}")
    print(f"  {'Random Forest':<20} {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.3f}  {test_auc_rf:.4f}")
    
    best_model_name = 'logistic_regression' if test_auc_lr > test_auc_rf else 'random_forest'
    best_auc = max(test_auc_lr, test_auc_rf)
    
    print(f"\n[BEST MODEL] Best model: {best_model_name.replace('_', ' ').title()} (Test AUC: {best_auc:.4f})")
    
    logger.info(f"Predictive modeling completed. Best AUC: {best_auc:.4f}")
    
    return {
        'models': models,
        'best_model': best_model_name,
        'best_auc': best_auc,
        'feature_columns': feature_columns,
        'scaler': scaler,
        'test_metrics': {
            'logistic_regression': {'auc': test_auc_lr},
            'random_forest': {'auc': test_auc_rf}
        }
    }
