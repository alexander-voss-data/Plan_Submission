"""
Main Trial Activation Analysis Script

This is the main orchestrator script that imports and runs the complete analysis
using the modular components. This script replaces the original monolithic file
and provides a clean, maintainable structure.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# Import custom modules
from config import *
from display_utils import *
from data_utils import *
from data_cleaner import *
from analysis_utils import *
from viz_utils import *
from statistical_utils import *

def setup_logging():
    """Set up logging configuration for the analysis."""
    log_filename = os.path.join(OUTPUT_DIR, f"trial_activation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main analysis function that orchestrates the entire process."""
    
    # Initialize logging and utilities
    logger = setup_logging()
    logger.info("Starting Trial Activation Analysis")
    
    # Create utility functions
    utils = create_utility_functions()
    
    # Set up plotting directories
    bivariate_dir, improved_dir = setup_plot_directories()
    
    # Display environment information
    print_phase_header("SETUP", "Environment and Configuration")
    logger.info("Environment setup completed successfully")
    logger.info(f"Random state set to: {RANDOM_STATE}")
    
    display_library_versions()
    
    errors = []
    
    try:
        # =============================================================================
        # Phase 1: Data Loading and Initial Setup
        # =============================================================================
        
        print_phase_header(1, "Data Loading and Initial Setup")
        print_step_header("1.1", "Environment Setup")
        
        print("[COMPLETED] Environment setup completed")
        print("[COMPLETED] Utility functions created")
        print("[COMPLETED] Plotting directories configured")
        
        print_step_header("1.2", "Data Loading and Basic Inspection")
        
        # Load the data
        df_raw = load_and_inspect_data(DATA_FILE)
        
        # Display basic information
        display_basic_info(df_raw)
        
        log_phase_completion(1, "Data Loading and Initial Setup", {
            "Records loaded": len(df_raw),
            "Columns": df_raw.shape[1],
            "Organizations": df_raw['ORGANIZATION_ID'].nunique()
        })
        
        # =============================================================================
        # Phase 2: Data Cleaning and Quality Assessment
        # =============================================================================
        
        print_phase_header(2, "Data Cleaning and Quality Assessment")
        
        print_step_header("2.1", "Duplicate Removal")
        df_clean = identify_and_remove_duplicates(df_raw, utils)
        
        print_step_header("2.2", "Data Type Conversion and Validation")
        df_typed, type_errors = convert_and_validate_data_types(df_clean)
        if type_errors:
            errors.append({"type": "Data Type Conversion", "message": str(type_errors)})
        
        print_step_header("2.3", "Missing Value Analysis")
        missing_analysis = analyze_missing_values(df_typed, utils)
        document_business_logic(df_typed, utils)
        
        print_step_header("2.4", "Data Validation Rules")
        validation_results, anomalies = validate_data_rules(df_typed, utils)
        if anomalies:
            errors.append({"type": "Data Validation", "message": f"{len(anomalies)} anomalous records found"})
        
        print_step_header("2.5", "Statistical Robustness Filtering")
        df_clean_final = apply_statistical_filtering(df_typed)
        
        # Save cleaned data
        output_file = os.path.join(OUTPUT_DIR, "data_clean.csv")
        df_clean_final.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to: {output_file}")
        
        log_phase_completion(2, "Data Cleaning and Quality Assessment", {
            "Original records": len(df_raw),
            "Final records": len(df_clean_final),
            "Organizations": df_clean_final['ORGANIZATION_ID'].nunique(),
            "Data reduction": f"{((len(df_raw) - len(df_clean_final))/len(df_raw)*100):.1f}%"
        })
        
        # =============================================================================
        # Phase 3: Feature Engineering
        # =============================================================================
        
        print_phase_header(3, "Feature Engineering")
        
        print_step_header("3.1", "Row-Level Temporal Features")
        df_with_temporal = create_temporal_features(df_clean_final)
        
        print_step_header("3.2", "Activity Categorization")
        df_with_categories = create_activity_categories(df_with_temporal, utils)
        
        print_step_header("3.3", "Organization-Level Aggregated Features")
        org_features_df = create_organization_level_features(df_with_categories)
        
        log_phase_completion(3, "Feature Engineering", {
            "Temporal features": "Created",
            "Activity categories": "Created", 
            "Organization features": len(org_features_df)
        })
        
        # =============================================================================
        # Phase 4: Dataset Segmentation and Analysis
        # =============================================================================
        
        print_phase_header(4, "Dataset Segmentation and Analysis")
        
        print_step_header("4.1", "Dataset Segmentation")
        segment_not_converted, segment_pre_conversion, segment_all_converted = create_dataset_segments(df_with_categories)
        
        print_step_header("4.2", "Univariate Analysis")
        univariate_results = perform_univariate_analysis(segment_not_converted, segment_pre_conversion, segment_all_converted)
        
        print_step_header("4.3", "Comparative Bivariate Analysis")
        bivariate_results = perform_bivariate_analysis(segment_not_converted, segment_pre_conversion, org_features_df)
        
        print_step_header("4.4", "Advanced EDA and Pattern Discovery")
        eda_results = perform_advanced_eda(segment_not_converted, segment_pre_conversion, org_features_df)
        
        print_step_header("4.5", "Basic Visualization")
        
        # Create and save basic visualizations
        logger.info("Creating visualization plots")
        
        # Activity distribution plot
        fig1 = create_activity_distribution_plot(df_with_categories, title="Overall Activity Distribution")
        save_plot(fig1, "activity_distribution.png", bivariate_dir)
        plt.close(fig1)
        
        # Category distribution plot
        fig2 = create_category_distribution_plot(df_with_categories, title="Activity Category Distribution")
        save_plot(fig2, "category_distribution.png", bivariate_dir)
        plt.close(fig2)
        
        # Temporal analysis plot
        fig3 = create_temporal_analysis_plot(df_with_categories, title="Temporal Activity Patterns")
        save_plot(fig3, "temporal_activity_analysis.png", bivariate_dir)
        plt.close(fig3)
        
        # Conversion comparison plot (using organization-level features)
        converted_org_features = org_features_df[org_features_df['converted'] == True]
        non_converted_org_features = org_features_df[org_features_df['converted'] == False]
        
        comparison_features = ['total_activities_count', 'unique_activities_count', 
                             'unique_categories_count', 'week1_activity_count',
                             'week1_activity_ratio', 'activities_per_day']
        
        fig4 = create_conversion_comparison_plot(
            converted_org_features, non_converted_org_features, 
            comparison_features, title="Converted vs Non-Converted Organizations"
        )
        save_plot(fig4, "conversion_comparison.png", improved_dir)
        plt.close(fig4)
        
        # Feature correlation heatmap
        numeric_features = ['total_activities_count', 'unique_activities_count', 
                          'activities_per_day', 'unique_categories_count', 'days_active',
                          'week1_activity_count', 'week1_activity_ratio']
        
        fig5 = create_correlation_heatmap(org_features_df, numeric_features, 
                                        title="Organization Feature Correlations")
        save_plot(fig5, "feature_correlation_analysis.png", bivariate_dir)
        plt.close(fig5)
        
        log_phase_completion(4, "Dataset Segmentation and Analysis", {
            "Segments created": 3,
            "Univariate analysis": "Completed",
            "Bivariate analysis": "Completed", 
            "Advanced EDA": "Completed",
            "Visualizations": 5,
            "Plots saved": f"{bivariate_dir}, {improved_dir}"
        })
        
        # =============================================================================
        # Phase 5: Statistical Testing and Predictive Modeling
        # =============================================================================
        
        print_phase_header(5, "Statistical Testing and Predictive Modeling")
        
        print_step_header("5.1", "Hypothesis Testing")
        statistical_results = perform_focused_conversion_hypothesis_tests(
            org_features_df, segment_not_converted, segment_pre_conversion
        )
        
        print_step_header("5.2", "Predictive Modeling")
        modeling_results = build_predictive_models(org_features_df)
        
        log_phase_completion(5, "Statistical Testing and Predictive Modeling", {
            "Statistical tests": len(statistical_results),
            "Significant tests": len([r for r in statistical_results.values() if r.get('significant', False)]),
            "Best model AUC": f"{modeling_results['best_auc']:.4f}",
            "Best model": modeling_results['best_model']
        })
        
        # =============================================================================
        # Final Summary
        # =============================================================================
        
        # Create final summary report
        analysis_results = {
            "Total organizations analyzed": org_features_df.shape[0],
            "Conversion rate": f"{(org_features_df['converted'].mean() * 100):.2f}%",
            "Statistical tests performed": len(statistical_results),
            "Significant findings": len([r for r in statistical_results.values() if r.get('significant', False)]),
            "Best model AUC": f"{modeling_results['best_auc']:.4f}",
            "Best model type": modeling_results['best_model'].replace('_', ' ').title(),
            "Key insight": "Complete analysis with statistical validation and predictive modeling"
        }
        
        create_summary_report(df_clean_final, analysis_results)
        
        # Print error summary
        print_error_summary(errors)
        
        print(f"\n[SUCCESS] Analysis completed successfully!")
        print(f"[OUTPUT] Output files saved in: {OUTPUT_DIR}/")
        print(f"[PLOTS] Plots saved in: {bivariate_dir}/ and {improved_dir}/")
        
        logger.info("Trial Activation Analysis completed successfully")
        
        return {
            'raw_data': df_raw,
            'clean_data': df_clean_final,
            'categorized_data': df_with_categories,
            'org_features': org_features_df,
            'segments': {
                'not_converted': segment_not_converted,
                'pre_conversion': segment_pre_conversion,
                'all_converted': segment_all_converted
            },
            'analysis_results': {
                'univariate': univariate_results,
                'bivariate': bivariate_results,
                'eda': eda_results
            },
            'statistical_results': statistical_results,
            'modeling_results': modeling_results,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the main analysis
    results = main()
