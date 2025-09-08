## TASK 1
Explore and clean the data set with Python. Using (advanced) analytics practices, discover what activities are indicative for conversion and defi ne trial goals. Describe your approach.


## DETAILED IMPLEMENTATION PLAN FOR TASK 1

### Phase 1: Data Loading and Initial Setup
**Objective**: Set up environment and load data with proper data types

#### Step 1.1: Environment Setup
- Import required libraries: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, scipy
- Set up logging and configuration
- Create utility functions for consistent formatting

#### Step 1.2: Data Loading and Basic Inspection
- Load CSV with proper data types
- Display basic dataset information (shape, columns, dtypes)
- Check file size and memory usage
- Preview first/last 10 rows

### Phase 2: Data Cleaning and Quality Assessment
**Objective**: Ensure data quality and consistency

#### Step 2.1: Duplicate Removal
- Identify duplicates based on (ORGANIZATION_ID + ACTIVITY_NAME + TIMESTAMP)
- Document duplicate count and patterns
- Remove duplicates and log impact

#### Step 2.2: Data Type Conversion and Validation
- Convert TIMESTAMP, CONVERTED_AT, TRIAL_START, TRIAL_END to datetime
- Validate timezone consistency
- Handle any parsing errors
- Ensure CONVERTED column is boolean

#### Step 2.3: Missing Value Analysis
- Generate missing value report by column
- Analyze patterns in CONVERTED_AT missingness
- Document business logic for missing values
- No imputation needed (missing CONVERTED_AT = not converted)

#### Step 2.4: Data Validation Rules
- Verify TIMESTAMP falls between TRIAL_START and TRIAL_END
- Check that CONVERTED_AT ≤ TRIAL_END when not null
- Validate trial duration is 30 days
- Flag and document any anomalies

### Phase 3: Feature Engineering
**Objective**: Create meaningful features for analysis

#### Step 3.1: Row-Level Temporal Features
- **trial_duration_days**: (TRIAL_END - TRIAL_START).days
- **days_since_trial_start**: (TIMESTAMP - TRIAL_START).days
- **days_until_trial_end**: (TRIAL_END - TIMESTAMP).days
- **hour**: TIMESTAMP.hour
- **day_of_week**: TIMESTAMP.dayofweek
- **is_weekend**: Boolean flag

#### Step 3.2: Activity Categorization
Create activity_category based on Attachment 1:
- 'Scheduling_Shifts': Scheduling and shift-related activities
- 'Absence_Management': Absence requests and approvals
- 'Time_Tracking': Punch clock and break activities
- 'Approvals_Payroll': Approval and payroll activities
- 'Financials': Revenue and budget activities
- 'Communication': Message and communication activities

#### Step 3.3: Organization-Level Aggregated Features
For each ORGANIZATION_ID, calculate:
- **unique_activities_count**: Number of distinct activities
- **total_activities_count**: Total activity count
- **activities_per_day**: total_activities / trial_duration
- **unique_categories_count**: Number of distinct activity categories used
- **days_active**: Number of unique days with activities
- **activity_density**: activities_per_day / days_active
- **time_to_first_activity**: Hours from trial start to first activity
- **activity_spread**: Standard deviation of days_since_trial_start

### Phase 4: Dataset Segmentation and Exploratory Data Analysis
**Objective**: Segment data early and understand conversion patterns through comparative analysis

#### Step 4.1: Dataset Segmentation (Moved from Phase 5)
Create three analysis segments for all subsequent analysis:
1. **Non-converted organizations**: CONVERTED = False (all activities)
2. **Pre-conversion activities**: Only activities before CONVERTED_AT for converted orgs
3. **All converted organization activities**: Complete activity history for converted orgs (for reference)

**Primary Focus**: Compare segments 1 and 2 to identify conversion indicators

#### Step 4.2: Univariate Analysis
- **Conversion rate**: Overall and by time period (trial start month/week cohorts)
- **Activity distribution**: Count and frequency of each activity (segments 1 vs 2)
- **Category distribution**: Usage of different activity categories (segments 1 vs 2)
- **Temporal patterns**: Activity volume by hour, day of week (segments 1 vs 2)
- **Trial duration validation**: Confirm 30-day trials

#### Step 4.3: Comparative Bivariate Analysis (Segments 1 vs 2)
**Core Analysis**: Direct comparison between non-converted and pre-conversion activities

- **Activity Occurrence Histograms**: 
  - Total activity counts per organization (segment 1 vs 2)
  - Unique activity counts per organization (segment 1 vs 2)
  - Activities per category per organization (segment 1 vs 2)

- **Temporal Occurrence Patterns**:
  - Activity frequency by days_since_trial_start (segment 1 vs 2)
  - Activity frequency by hour of day (segment 1 vs 2) 
  - Activity frequency by day of week (segment 1 vs 2)
  - Time-to-first-activity distributions (segment 1 vs 2)

- **Activity-Specific Comparisons**:
  - Occurrence rate of each individual activity (segment 1 vs 2)
  - Occurrence rate by activity category (segment 1 vs 2)
  - Activity penetration rates (% of orgs using each activity)
  - Activity intensity (average count per org that uses it)

#### Step 4.4: Advanced EDA and Pattern Discovery
- **Cohort analysis**: Conversion rates by trial start week/month
- **Activity sequence analysis**: Common activity progressions leading to conversion
- **Time-to-event analysis**: Distribution of conversion timing within trial period
- **Activity clustering**: Groups of activities commonly used together
- **Temporal conversion patterns**: When during trials do conversions typically occur

### Phase 5: Statistical Testing and Model Validation
**Objective**: Validate findings from comparative analysis with statistical rigor and leverage early/late stage timing insights

#### Step 5.1: Enhanced Feature Engineering (Using Segmented Data)
Working with segments 1 and 2 from Phase 4 with focus on **early/late stage patterns**:

**Time-Based Features**:
- **early_stage_activity_count**: Total activities in first 2 weeks (days 0-13)
- **late_stage_activity_count**: Total activities in last 2 weeks (days 14-27)
- **early_stage_activity_ratio**: Early stage activities as % of total trial activities  
- **late_stage_activity_ratio**: Late stage activities as % of total trial activities
- **early_stage_engagement_binary**: Binary indicator for early stage activity above threshold
- **late_stage_engagement_binary**: Binary indicator for late stage activity above threshold
- **early_vs_late_activity_ratio**: Early stage activities / Late stage activities
- **early_dominated_pattern**: Early stage activity ratio > 60% (binary)
- **balanced_engagement_pattern**: Early stage ratio between 30-70% (binary)

**Individual Activity Features**:
- **Activity adoption**: Binary indicators for usage of key individual activities
- **Activity usage intensity**: Count of usage for each key activity
- **Early stage activity adoption**: Usage of individual activities in first 2 weeks
- **Late stage activity adoption**: Usage of individual activities in last 2 weeks
- **Heavy user patterns**: Binary indicators for high-frequency usage of activities

**Combined Features**:
- **Early intensity per activity**: Early stage count × individual activity usage
- **Late intensity per activity**: Late stage count × individual activity usage  
- **Consistent engagement per activity**: Usage in both early and late stages

**Traditional Activity Features**:
- **Activity density**: Activities per active day
- **Time-to-first-activity**: Hours from trial start to first activity
- **Activity spread**: Standard deviation of activity timing
- **Unique activities/categories**: Counts of distinct activities and categories used

#### Step 5.2: Early/Late Stage Pattern Statistical Testing
**Primary Hypothesis**: Early stage engagement patterns combined with individual activity adoption are strongest predictors of conversion

**Early/Late Stage-Focused Tests**:
- **Early Stage Activity Volume**: T-test/Mann-Whitney U on early stage activity counts
- **Early Dominance Pattern**: Chi-square test on early vs late stage engagement distribution
- **Stage Progression**: ANOVA on early-to-late stage activity changes
- **Early Stage Engagement Rate**: Chi-square test on organizations active in early stages

**Individual Activity Tests**:
- **Activity adoption patterns**: Chi-square tests on individual activity adoption rates
- **Usage intensity differences**: T-tests on activity usage counts between segments
- **Stage-specific adoption**: Tests on early vs late stage adoption patterns
- **Combined pattern analysis**: Tests on early/late + individual activity combinations

**Traditional Statistical Tests**:
- **Chi-square tests**: Activity presence differences (segment 1 vs 2)
- **T-tests/Mann-Whitney U**: Continuous feature differences (counts, timing, velocity)
- **Multiple hypothesis correction**: Bonferroni or FDR correction for multiple comparisons
- **Effect size calculation**: Cohen's d for practical significance
- **Kolmogorov-Smirnov tests**: Distribution differences for key metrics

#### Step 5.3: Advanced Statistical Modeling to Achieve AUC ≥ 0.7



## TECHNICAL STRUCTURE OF IMPLEMENTATION

## File Structure

```
Task1/
├── main_analysis.py           # Main script - run this to execute the analysis
├── config.py                  # Configuration settings and constants
├── data_utils.py             # Data loading and inspection functions
├── data_cleaner.py           # Data cleaning and validation functions
├── analysis_utils.py         # Feature engineering and analysis functions
├── viz_utils.py              # Visualization and plotting functions
├── display_utils.py          # Formatting and display utilities
├── bivariate_analysis_plots/ # Output directory for analysis plots
├── improved_activity_plots/  # Output directory for improved plots
└── logs/                     # Analysis log files (auto-created)
```


### Requirements
- Python 3.7+
- Core libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn 
- Statistical libraries: statsmodels
- Optional libraries: plotly, xgboost (for extended analysis)
- Standard library: logging, os, datetime

## Module Descriptions

### `main_analysis.py` - Main Orchestrator
- Entry point for the entire analysis
- Coordinates all phases of the analysis
- Handles error management and logging
- Generates final summary reports

### `config.py` - Configuration
- All constants and configuration settings
- Plot styling and output directories
- Analysis parameters (thresholds, random seeds)
- Easy to modify without touching analysis code

### `data_utils.py` - Data Loading
- `load_and_inspect_data()` - Load CSV and perform initial inspection
- `display_basic_info()` - Comprehensive dataframe information display
- Handles file I/O errors gracefully

### `data_cleaner.py` - Data Cleaning
- `identify_and_remove_duplicates()` - Remove duplicate records
- `convert_and_validate_data_types()` - Convert and validate data types
- `analyze_missing_values()` - Analyze missing value patterns
- `validate_data_rules()` - Apply business rule validation
- `apply_statistical_filtering()` - Remove statistical outliers

### `analysis_utils.py` - Feature Engineering & Analysis
- `create_temporal_features()` - Generate time-based features
- `create_activity_categories()` - Categorize activities by business function
- `create_organization_level_features()` - Aggregate features per organization
- `create_dataset_segments()` - Segment data for comparative analysis

### `viz_utils.py` - Visualizations
- `create_activity_distribution_plot()` - Activity distribution charts
- `create_category_distribution_plot()` - Category distribution charts
- `create_temporal_analysis_plot()` - Time-based analysis plots
- `create_conversion_comparison_plot()` - Compare converted vs non-converted
- `create_correlation_heatmap()` - Feature correlation analysis
- `setup_plot_directories()` - Manage output directories

### `display_utils.py` - Display & Formatting
- `print_phase_header()` / `print_step_header()` - Consistent formatting
- `create_utility_functions()` - Number formatting, percentages, etc.
- `display_library_versions()` - For reproducibility
- `create_summary_report()` - Generate comprehensive reports



##  Output Files

The analysis generates the following outputs in the `Task1/` directory:

### Data Files
- `data_clean.csv` - Cleaned and processed dataset

### Visualizations
- `bivariate_analysis_plots/` - Analysis visualizations
- `improved_activity_plots/` - Enhanced activity plots

### Logs
- `trial_activation_analysis_YYYYMMDD_HHMMSS.log` - Detailed execution log

