"""
Task 2: SQL-Based Data Mart Layer for Trial Activation Tracking
==============================================================

Implementation of SQL-based data mart layer with staging and mart views
to track trial activation using 3 key metrics:
1. Activity in first week
2. Activity in second week  
3. Steadiness of activities in week 4

Author: Alexander Voß
Date: September 8, 2025
"""

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import os
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set export directory for CSV exports to the current Task2 folder
EXPORT_DIR = os.path.dirname(__file__)


class TrialActivationDataMart:
    """
    SQL-based data mart for trial activation tracking.
    
    Architecture: Raw Data (CSV) → Staging Layer → Mart Layer (Views) → CSV Export
    """
    
    def __init__(self, csv_path: str, output_dir: str = "."):
        """
        Initialize the data mart processor.
        
        Args:
            csv_path: Path to the source CSV file
            output_dir: Directory for CSV exports
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.engine = create_engine('sqlite:///:memory:', echo=False)
        self.connection = self.engine.connect()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized data mart with source: {csv_path}")
        logger.info(f"Output directory: {output_dir}")

    def load_and_prepare_data(self):
        """
        Phase 1: Data Ingestion - Load and prepare raw data with derived fields.
        """
        logger.info("Phase 1: Loading and preparing raw data...")
        
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Convert datetime columns
        datetime_columns = ['TIMESTAMP', 'TRIAL_START', 'TRIAL_END', 'CONVERTED_AT']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate derived fields
        df['days_since_trial_start'] = (df['TIMESTAMP'] - df['TRIAL_START']).dt.days
        df['trial_week'] = ((df['days_since_trial_start']) // 7) + 1
        df['trial_week'] = df['trial_week'].clip(1, 4)  # Clip to weeks 1-4
        
        # Handle edge cases
        df['days_since_trial_start'] = df['days_since_trial_start'].fillna(0)
        df['trial_week'] = df['trial_week'].fillna(1).astype(int)
        
        self.raw_df = df
        logger.info("✓ Data preparation completed")
        logger.info(f"  - Date conversion completed for {len(datetime_columns)} columns")
        logger.info(f"  - Trial week calculation completed")
        logger.info(f"  - Trial week distribution: {df['trial_week'].value_counts().sort_index().to_dict()}")
        
        return df

    def create_staging_metrics_per_org(self):
        """
        Phase 2A: Create staging_metrics_per_org table with organization-level metrics.
        """
        logger.info("Phase 2A: Creating staging_metrics_per_org table...")
        
        df = self.raw_df
        
        # Calculate weekly activity counts per organization
        weekly_activities = df.groupby(['ORGANIZATION_ID', 'trial_week']).size().reset_index(name='activity_count')
        
        # Pivot to get week columns
        weekly_pivot = weekly_activities.pivot(
            index='ORGANIZATION_ID', 
            columns='trial_week', 
            values='activity_count'
        ).fillna(0).astype(int)
        
        # Ensure all week columns exist (1-4)
        for week in [1, 2, 3, 4]:
            if week not in weekly_pivot.columns:
                weekly_pivot[week] = 0
        
        # Rename columns for clarity
        weekly_pivot = weekly_pivot.rename(columns={
            1: 'week1_activities',
            2: 'week2_activities', 
            3: 'week3_activities',
            4: 'week4_activities'
        })
        
        # Calculate total activities per organization
        total_activities = df.groupby('ORGANIZATION_ID').size().reset_index(name='total_activities')
        
        # Get conversion status and trial dates
        org_metadata = df.groupby('ORGANIZATION_ID').agg({
            'CONVERTED': 'first',
            'TRIAL_START': 'first',
            'TRIAL_END': 'first'
        }).reset_index()
        
        # Combine all metrics
        staging_metrics = weekly_pivot.reset_index()
        staging_metrics = staging_metrics.merge(total_activities, on='ORGANIZATION_ID', how='left')
        staging_metrics = staging_metrics.merge(org_metadata, on='ORGANIZATION_ID', how='left')
        
        # Rename and format columns
        staging_metrics = staging_metrics.rename(columns={'ORGANIZATION_ID': 'org_id'})
        staging_metrics['converted'] = staging_metrics['CONVERTED'].astype(bool)
        staging_metrics['trial_start_date'] = staging_metrics['TRIAL_START'].dt.date
        staging_metrics['trial_end_date'] = staging_metrics['TRIAL_END'].dt.date
        
        # Select final columns
        final_columns = [
            'org_id', 'week1_activities', 'week2_activities', 'week3_activities', 'week4_activities',
            'total_activities', 'converted', 'trial_start_date', 'trial_end_date'
        ]
        staging_metrics = staging_metrics[final_columns]
        
        # Write to SQLite table
        staging_metrics.to_sql('staging_metrics_per_org', self.connection, if_exists='replace', index=False)
        
        logger.info("✓ staging_metrics_per_org table created successfully")
        logger.info(f"  - Records: {len(staging_metrics)}")
        logger.info(f"  - Columns: {list(staging_metrics.columns)}")
        logger.info(f"  - Converted organizations: {staging_metrics['converted'].sum()}")
        
        # Store for threshold calculation
        self.staging_metrics = staging_metrics
        
        return staging_metrics

    def create_staging_activation_goals(self):
        """
        Phase 2B: Create staging_activation_goals table with configurable thresholds.
        """
        logger.info("Phase 2B: Creating staging_activation_goals table...")
        
        # Calculate 75th percentile thresholds from converted organizations
        converted_orgs = self.staging_metrics[self.staging_metrics['converted'] == True]
        
        if len(converted_orgs) == 0:
            logger.warning("No converted organizations found, using default thresholds")
            week1_threshold = 10.0  # Default fallback
            minimal_events_threshold = 3.0   # Default baseline for modeling
        else:
            week1_threshold = converted_orgs['week1_activities'].quantile(0.75)
            minimal_events_threshold = 3.0  # Fixed threshold for minimal events as modeling basis
        
        # Create goals data
        goals_data = [
            {
            'goal_id': 'WEEK1_MIN_ACTIVITY',
            'metric_name': 'Week 1 Minimum Activities',
            'threshold_value': float(week1_threshold),
            'description': f'Minimum activities required in first week (75th percentile of converted orgs: {week1_threshold:.1f})'
            },
            {
            'goal_id': 'MINIMAL_EVENTS_THRESHOLD', 
            'metric_name': 'Minimal Events (Modeling Basis)',
            'threshold_value': float(minimal_events_threshold),
            'description': f'Minimal total events required as basis for modeling (fixed threshold: {minimal_events_threshold:.0f})'
            },
            {
            'goal_id': 'WEEK4_STEADINESS_RATIO',
            'metric_name': 'Week 4 Steadiness Ratio', 
            'threshold_value': 0.8,
            'description': 'Week 4 activities must be ≥80% of Week 1 activities (fixed business rule)'
            }
        ]
        
        # Create DataFrame and write to SQLite
        staging_goals = pd.DataFrame(goals_data)
        staging_goals.to_sql('staging_activation_goals', self.connection, if_exists='replace', index=False)
        
        # Get the steadiness ratio for use in views
        week4_steadiness_ratio = goals_data[2]['threshold_value']
        
        logger.info("✓ staging_activation_goals table created successfully")
        logger.info(f"  - Goals defined: {len(staging_goals)}")
        logger.info(f"  - Week 1 threshold: {week1_threshold:.1f} activities")
        logger.info(f"  - Minimal events threshold: {minimal_events_threshold:.0f} activities")
        logger.info(f"  - Week 4 steadiness ratio: {week4_steadiness_ratio} ({week4_steadiness_ratio*100}%)")
        
        # Store thresholds for use in views
        self.week1_threshold = week1_threshold
        self.minimal_events_threshold = minimal_events_threshold
        self.week4_steadiness_ratio = week4_steadiness_ratio
        
        return staging_goals
    
    def create_mart_trial_goals_view(self):
        """
        Phase 3A: Create mart_trial_goals view with completion percentages.
        """
        logger.info("Phase 3A: Creating mart_trial_goals view...")
        
        # Create the view with proper SQL logic using variable ratio
        view_sql = text("""
        CREATE VIEW mart_trial_goals AS
        SELECT 
            m.org_id,
            m.week1_activities,
            m.week2_activities,
            m.week4_activities,
            m.total_activities,
            m.converted,
            m.trial_start_date,
            m.trial_end_date,
            
            -- Week 1 completion percentage
            CASE 
            WHEN m.week1_activities >= g1.threshold_value THEN 100.0
            WHEN g1.threshold_value = 0 THEN 100.0
            ELSE (CAST(m.week1_activities AS FLOAT) / g1.threshold_value) * 100.0 
            END as week1_completion_pct,
            
            -- Minimal events completion percentage (based on total activities)
            CASE 
            WHEN m.total_activities >= g2.threshold_value THEN 100.0
            WHEN g2.threshold_value = 0 THEN 100.0
            ELSE (CAST(m.total_activities AS FLOAT) / g2.threshold_value) * 100.0 
            END as minimal_events_completion_pct,
            
            -- Week 4 steadiness target calculation (based on Week 1 and variable ratio)
            m.week1_activities * g3.threshold_value as week4_target_activities,
            
            -- Week 4 completion percentage (based on Week 1 and variable ratio)
            CASE 
            WHEN m.week4_activities >= (m.week1_activities * g3.threshold_value) THEN 100.0
            WHEN (m.week1_activities * g3.threshold_value) = 0 THEN 100.0
            ELSE (CAST(m.week4_activities AS FLOAT) / (m.week1_activities * g3.threshold_value)) * 100.0 
            END as week4_completion_pct
            
        FROM staging_metrics_per_org m
        CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'WEEK1_MIN_ACTIVITY') g1
        CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'MINIMAL_EVENTS_THRESHOLD') g2
        CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'WEEK4_STEADINESS_RATIO') g3
        """)
        
        self.connection.execute(view_sql)
        
        # Add overall_achieved calculation in a separate view update
        # (SQLite doesn't support computed columns referring to other computed columns in same SELECT)
        # First drop the existing view, then create the new one
        try:
            self.connection.execute(text("DROP VIEW mart_trial_goals"))
        except:
            pass  # View might not exist yet
        
        view_with_achievement_sql = text("""
        CREATE VIEW mart_trial_goals AS
        SELECT *,
            CASE 
                WHEN week1_completion_pct >= 100 AND 
                     minimal_events_completion_pct >= 100 AND 
                     week4_completion_pct >= 100 THEN 1
                ELSE 0 
            END as overall_achieved
        FROM (
            SELECT 
                m.org_id,
                m.week1_activities,
                m.week2_activities,
                m.week4_activities,
                m.total_activities,
                m.converted,
                m.trial_start_date,
                m.trial_end_date,
                
                -- Week 1 completion percentage
                CASE 
                    WHEN m.week1_activities >= g1.threshold_value THEN 100.0
                    WHEN g1.threshold_value = 0 THEN 100.0
                    ELSE (CAST(m.week1_activities AS FLOAT) / g1.threshold_value) * 100.0 
                END as week1_completion_pct,
                
                -- Minimal events completion percentage  
                CASE 
                    WHEN m.total_activities >= g2.threshold_value THEN 100.0
                    WHEN g2.threshold_value = 0 THEN 100.0
                    ELSE (CAST(m.total_activities AS FLOAT) / g2.threshold_value) * 100.0 
                END as minimal_events_completion_pct,
                
                -- Week 4 steadiness target calculation (based on Week 1)
                m.week1_activities * g3.threshold_value as week4_target_activities,

                -- Week 4 completion percentage (based on Week 1)
                CASE 
                    WHEN m.week4_activities >= (m.week1_activities * g3.threshold_value) THEN 100.0
                    WHEN (m.week1_activities * g3.threshold_value) = 0 THEN 100.0
                    ELSE (CAST(m.week4_activities AS FLOAT) / (m.week1_activities * g3.threshold_value)) * 100.0 
                END as week4_completion_pct
                
            FROM staging_metrics_per_org m
            CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'WEEK1_MIN_ACTIVITY') g1
            CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'MINIMAL_EVENTS_THRESHOLD') g2
            CROSS JOIN (SELECT threshold_value FROM staging_activation_goals WHERE goal_id = 'WEEK4_STEADINESS_RATIO') g3
            WHERE m.converted = 0
        ) subquery
        """)
        
        self.connection.execute(view_with_achievement_sql)
        
        logger.info("✓ mart_trial_goals view created successfully")
        
        # Test the view
        test_query = pd.read_sql("SELECT COUNT(*) as record_count FROM mart_trial_goals", self.connection)
        logger.info(f"  - Records in view: {test_query['record_count'].iloc[0]}")
        
        # Check achievement statistics
        achievement_stats = pd.read_sql("""
            SELECT 
                SUM(overall_achieved) as activated_orgs,
                COUNT(*) as total_orgs,
                ROUND(AVG(week1_completion_pct), 1) as avg_week1_completion,
                ROUND(AVG(minimal_events_completion_pct), 1) as avg_minimal_events_completion,
                ROUND(AVG(week4_completion_pct), 1) as avg_week4_completion
            FROM mart_trial_goals
        """, self.connection)
        
        stats = achievement_stats.iloc[0]
        logger.info(f"  - Activated organizations: {stats['activated_orgs']} / {stats['total_orgs']} ({stats['activated_orgs']/stats['total_orgs']*100:.1f}%)")
        logger.info(f"  - Average completion rates: Week1={stats['avg_week1_completion']}%, MinimalEvents={stats['avg_minimal_events_completion']}%, Week4={stats['avg_week4_completion']}%")

    def create_mart_trial_activation_view(self):
        """
        Phase 3B: Create mart_trial_activation view with only activated organizations.
        """
        logger.info("Phase 3B: Creating mart_trial_activation view...")
        
        view_sql = text("""
        CREATE VIEW mart_trial_activation AS
        SELECT *,
            -- Additional performance score
            ROUND((week1_completion_pct + minimal_events_completion_pct + week4_completion_pct) / 3.0, 2) as avg_performance_score
        FROM mart_trial_goals
        WHERE overall_achieved = 1
        """)
        
        self.connection.execute(view_sql)
        
        logger.info("✓ mart_trial_activation view created successfully")
        
        # Test the view
        test_query = pd.read_sql("SELECT COUNT(*) as activated_count FROM mart_trial_activation", self.connection)
        activated_count = test_query['activated_count'].iloc[0]
        logger.info(f"  - Activated organizations: {activated_count}")
        
        if activated_count > 0:
            performance_stats = pd.read_sql("""
                SELECT 
                    ROUND(AVG(avg_performance_score), 1) as avg_performance,
                    MIN(avg_performance_score) as min_performance,
                    MAX(avg_performance_score) as max_performance
                FROM mart_trial_activation
            """, self.connection)
            
            stats = performance_stats.iloc[0]
            logger.info(f"  - Performance scores: Avg={stats['avg_performance']}, Min={stats['min_performance']}, Max={stats['max_performance']}")

    def export_all_tables(self):
        """
        Phase 4: Export all staging tables and mart views to CSV files.
        """
        logger.info("Phase 4: Exporting all tables and views to CSV...")
        
        # Define tables/views to export
        exports = [
            'staging_metrics_per_org',
            'staging_activation_goals', 
            'mart_trial_goals',
            'mart_trial_activation'
        ]
        
        export_results = {}
        
        for table_name in exports:
            try:
                # Read data from SQLite
                df = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                
                # Export to CSV
                csv_path = os.path.join(EXPORT_DIR, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                
                export_results[table_name] = {
                    'records': len(df),
                    'columns': len(df.columns),
                    'file_path': csv_path
                }
                
                logger.info(f"  ✓ {table_name}: {len(df)} records → {csv_path}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to export {table_name}: {str(e)}")
                export_results[table_name] = {'error': str(e)}
        
        logger.info("✓ CSV export completed successfully")
        return export_results

    def run_data_quality_checks(self):
        """
        Run comprehensive data quality checks and validation.
        """
        logger.info("Running data quality checks...")
        
        checks = {}
        
        # Check 1: Record count validation
        staging_count = pd.read_sql("SELECT COUNT(*) as count FROM staging_metrics_per_org", self.connection)['count'].iloc[0]
        mart_count = pd.read_sql("SELECT COUNT(*) as count FROM mart_trial_goals", self.connection)['count'].iloc[0]
        checks['record_count_consistency'] = staging_count == mart_count
        logger.info(f"  Record count consistency: {'✓' if checks['record_count_consistency'] else '✗'} (Staging: {staging_count}, Mart: {mart_count})")
        
        # Check 2: Threshold validation
        goals = pd.read_sql("SELECT * FROM staging_activation_goals", self.connection)
        week1_threshold = goals[goals['goal_id'] == 'WEEK1_MIN_ACTIVITY']['threshold_value'].iloc[0]
        minimal_events_threshold = goals[goals['goal_id'] == 'MINIMAL_EVENTS_THRESHOLD']['threshold_value'].iloc[0]
        checks['thresholds_positive'] = week1_threshold > 0 and minimal_events_threshold > 0
        logger.info(f"  Threshold validation: {'✓' if checks['thresholds_positive'] else '✗'} (Week1: {week1_threshold}, MinimalEvents: {minimal_events_threshold})")
        
        # Check 3: Business logic validation (sample check)
        sample_check = pd.read_sql("""
            SELECT org_id, week1_activities, total_activities, week4_activities,
                   week1_completion_pct, minimal_events_completion_pct, week4_completion_pct, overall_achieved
            FROM mart_trial_goals 
            LIMIT 5
        """, self.connection)
        
        checks['sample_data_exists'] = len(sample_check) > 0
        logger.info(f"  Sample data validation: {'✓' if checks['sample_data_exists'] else '✗'}")
        
        # Check 4: View consistency
        activation_count = pd.read_sql("SELECT COUNT(*) as count FROM mart_trial_activation", self.connection)['count'].iloc[0]
        activated_in_goals = pd.read_sql("SELECT COUNT(*) as count FROM mart_trial_goals WHERE overall_achieved = 1", self.connection)['count'].iloc[0]
        checks['view_consistency'] = activation_count == activated_in_goals
        logger.info(f"  View consistency: {'✓' if checks['view_consistency'] else '✗'} (Activation view: {activation_count}, Goals achieved: {activated_in_goals})")
        
        # Overall validation result
        all_checks_passed = all(checks.values())
        logger.info(f"Overall data quality: {'✓ PASSED' if all_checks_passed else '✗ FAILED'}")
        
        return checks

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of the data mart.
        """
        logger.info("Generating summary report...")
        
        print("\n" + "="*80)
        print("TASK 2: SQL-BASED DATA MART LAYER - SUMMARY REPORT")
        print("="*80)
        
        # Basic statistics
        metrics = pd.read_sql("SELECT COUNT(*) as total_orgs FROM staging_metrics_per_org", self.connection)
        goals = pd.read_sql("SELECT COUNT(*) as total_goals FROM staging_activation_goals", self.connection)
        activated = pd.read_sql("SELECT COUNT(*) as activated_orgs FROM mart_trial_activation", self.connection)
        
        print(f"\nDATA MART STATISTICS:")
        print(f"   Total Organizations: {metrics['total_orgs'].iloc[0]}")
        print(f"   Activation Goals Defined: {goals['total_goals'].iloc[0]}")
        print(f"   Fully Activated Organizations: {activated['activated_orgs'].iloc[0]}")
        print(f"   Activation Rate: {activated['activated_orgs'].iloc[0] / metrics['total_orgs'].iloc[0] * 100:.1f}%")
        
        # Goal thresholds
        threshold_info = pd.read_sql("SELECT goal_id, threshold_value, description FROM staging_activation_goals", self.connection)
        print(f"\nACTIVATION THRESHOLDS:")
        for _, row in threshold_info.iterrows():
            print(f"   {row['goal_id']}: {row['threshold_value']}")
            print(f"      {row['description']}")
        
        # Completion statistics
        completion_stats = pd.read_sql("""
            SELECT
                ROUND(AVG(week1_completion_pct), 1) as avg_week1,
                ROUND(AVG(minimal_events_completion_pct), 1) as avg_minimal_events,
                ROUND(AVG(week4_completion_pct), 1) as avg_week4,
                COUNT(CASE WHEN week1_completion_pct >= 100 THEN 1 END) as week1_achieved,
                COUNT(CASE WHEN minimal_events_completion_pct >= 100 THEN 1 END) as minimal_events_achieved,
                COUNT(CASE WHEN week4_completion_pct >= 100 THEN 1 END) as week4_achieved,
                COUNT(*) as total
            FROM mart_trial_goals
        """, self.connection)
        
        stats = completion_stats.iloc[0]
        print(f"\nCOMPLETION ANALYSIS:")
        print(f"   Average Completion Rates:")
        print(f"      Week 1: {stats['avg_week1']}% ({stats['week1_achieved']}/{stats['total']} organizations)")
        print(f"      Minimal Events: {stats['avg_minimal_events']}% ({stats['minimal_events_achieved']}/{stats['total']} organizations)")  
        print(f"      Week 4: {stats['avg_week4']}% ({stats['week4_achieved']}/{stats['total']} organizations)")
        
        # Performance analysis for activated organizations
        if activated['activated_orgs'].iloc[0] > 0:
            perf_stats = pd.read_sql("""
                SELECT 
                    ROUND(AVG(avg_performance_score), 1) as avg_performance,
                    ROUND(MIN(avg_performance_score), 1) as min_performance,
                    ROUND(MAX(avg_performance_score), 1) as max_performance
                FROM mart_trial_activation
            """, self.connection)
            
            perf = perf_stats.iloc[0]
            print(f"\nACTIVATED ORGANIZATIONS PERFORMANCE:")
            print(f"   Average Performance Score: {perf['avg_performance']}%")
            print(f"   Performance Range: {perf['min_performance']}% - {perf['max_performance']}%")
        
        # Export summary
        print(f"\nEXPORTED FILES:")
        export_files = [
            'staging_metrics_per_org.csv',
            'staging_activation_goals.csv',
            'mart_trial_goals.csv', 
            'mart_trial_activation.csv'
        ]
        
        for file_name in export_files:
            file_path = os.path.join(self.output_dir, file_name)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✓ {file_name} ({file_size:,} bytes)")
            else:
                print(f"   ✗ {file_name} (missing)")
        
        print(f"\n✅ DATA MART IMPLEMENTATION COMPLETED SUCCESSFULLY")
        print("="*80)

    def run_complete_pipeline(self):
        """
        Execute the complete data mart pipeline from start to finish.
        """
        logger.info("Starting complete Task 2 data mart pipeline...")
        
        try:
            # Phase 1: Data Ingestion
            self.load_and_prepare_data()
            
            # Phase 2: Staging Layer
            self.create_staging_metrics_per_org()
            self.create_staging_activation_goals()
            
            # Phase 3: Mart Layer
            self.create_mart_trial_goals_view()
            self.create_mart_trial_activation_view()
            
            # Phase 4: CSV Export
            export_results = self.export_all_tables()
            
            # Validation
            quality_checks = self.run_data_quality_checks()
            
            # Summary Report
            self.generate_summary_report()
            
            logger.info("✅ Task 2 data mart pipeline completed successfully!")
            return {
                'status': 'success',
                'exports': export_results,
                'quality_checks': quality_checks
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
        
        finally:
            # Clean up connection
            self.connection.close()

def main():
    """
    Main execution function for Task 2.
    """
    # Configuration
    csv_path = "data_clean.csv"
    output_dir = "task2_outputs"
    
    print("="*80)
    print("TASK 2: SQL-BASED DATA MART LAYER FOR TRIAL ACTIVATION TRACKING")
    print("="*80)
    print(f"Source CSV: {csv_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run data mart
    data_mart = TrialActivationDataMart(csv_path, output_dir)
    results = data_mart.run_complete_pipeline()
    
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results

if __name__ == "__main__":
    main()
