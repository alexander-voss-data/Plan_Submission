## Trial Activation Senior Data Analyst Task
## Author: Alexander Voß, 7.9.2025


## Business Problem
The product team wants to improve the efficiency of converting trialists into paying customers by creating better user experiences. To measure their team’s success, they aim to track the "activation" of trialists based on specific actions that indicate a successful trial.


## Executive Summary
After cleaning the provided datasheet, I performed an extensive Explenatory Data Analysis (EDA) to come up with ideas on how to extend, analyse and modify the data in order to extract the required insights. My analysis showed, that early adoption of the tool can be a key driver of activation, alongside steady and prolonged use. 
The results of this analysis were used to create a SQL based model, which can be fed with live data while the thresholds for activiation can be set centrally. For reporting purposes, there are two tables (/views) which can be used to track activation success of individual trialist. 
Finally, I used the results of the EDA to create recommondations for the product teams to support their data-driven decision-making process.


## Insights & Results Task 1
After basic cleanup (removing duplicates), removal of outliers and data splitting (non converted vs converted before trial end), I performed EDA with Uni & Bivariate Analysis. The Bivariate Analysis Activity sequence suggests that there are differences in average actions per day & week. This could be supported by stastical test of hypothetises, which also revelead significant differences for day 2 and hourly activities. The latter 2 have been disregarded as it can be considered unlikely that there is a causal relationship between them and trial activation. EDA & stastistical testing also revealed, What's NOT significant:
  - Individual activity adoption rates
  - Organization-level aggregate features
  - Weekend vs weekday patterns
    
I tried to find patterns using advanced techniques like logistic regression,Random Forest, Decision Trees & XGBoost, without much success. So I leaned into Feature Engineering for time feature and interactions but I couldn't uncover more that the already suggested emphasis on early adoption.


## Insights & Results Task 2

Based on the insights from Task 1, I identified 3 main goals to track activation:
  - Minimum of 3 activities (because that was also the lower cut-off for the analysis)
  -  First week with at least 27 activities (75 percentile of converted orgs)
  -  Steady use throughout week 4 (altough sligthly lower: 80%)

Therefore, I created a data model based on 4 tables/views:
  1: staging table: metrics per org
  2: staging table: goals/thresholds
  3: *mart view: Trial Goals
  4: *mart view: Trial Activation
(*usually I would prefer to materialize mart layer table, but due to the setup inside pyhton I decided to create views)

The staging tables could be filled via different source systems. Metrics per org is derived from the analysis of task 1 while the goals are a direct result and could be maintained somewhere else or automatically calculated.
Based on this, the first mart view will track all open trialist and their progress towards the goals in %. All trialist which achieved at least 100% in all categories are shown in the final mart view: Trial Activation.

All tables/view were exported as views


## Insights & Results Task 3

Based on the analysis, I can provide the following insights which support the suggestions from task 1&2:

Conversion Metrics:
  - Converted organizations: 127
  - Total organizations: 599
  - Conversion rate: 21.20%

Key Weekly Activity Insights:
  - Week 1: Converted orgs: 39.3 avg activities/org vs Non-converted: 35.0 (+4.2 difference)
  - Week 2: Converted orgs: 88.1 avg activities/org vs Non-converted: 77.4 (+10.7 difference)
  - Week 3: Converted orgs: 108.9 avg activities/org vs Non-converted: 130.6 (-21.7 difference)
  - Week 4: Converted orgs: 105.7 avg activities/org vs Non-converted: 175.7 (-70.0 difference)

-> while converted orgs begin stronger, Non-converted increase steadily, which is an indication that there is demand which possibly can be converted to achieve higher conversion rate.

-> early & high quality onboarding can be a key success factor

Usually, I would include metrics like time-to-activate or Retation rate to such an analysis but the data does not seem to be suited for such metrics.


## Folder Structure

**You have to import Senior DA task (1).csv into the main forlder in order to make the scripts work**

```
Final/
├── data_clean.csv                    # Cleaned dataset used for analysis
├── Senior DA task (1).csv           # Original raw dataset
├── README.md                        # Main project documentation
├── Task1.md                         # Detailed Task 1 documentation 
├── Task2.md                         # Detailed Task 2 documentation 
├── Task3.md                         # Task 3 description 
├── Task1/                           # Modular Python analysis implementation
│   ├── main_analysis.py            # Main orchestrator script
│   ├── config.py                   # Configuration and settings
│   ├── data_utils.py               # Data loading utilities
│   ├── data_cleaner.py             # Data cleaning and validation
│   ├── analysis_utils.py           # Feature engineering and EDA
│   ├── viz_utils.py                # Visualization utilities
│   ├── display_utils.py            # Output formatting utilities
│   ├── statistical_utils.py       # Statistical testing and modeling
│   ├── trial_activation_analysis_*.log  # Analysis execution logs
│   ├── bivariate_analysis_plots/   # Bivariate analysis visualizations
│   └── improved_activity_plots/    # Activity and category analysis plots
├── Task2/                           # SQL data mart implementation
│   ├── task2_sql_data_mart.py      # SQL data mart creation script
│   └── task2_outputs/              # Generated data mart tables
│       ├── staging_metrics_per_org.csv     # Staging: Organization metrics
│       ├── staging_activation_goals.csv    # Staging: Activation thresholds
│       ├── mart_trial_goals.csv           # Mart: Trial progress tracking
│       └── mart_trial_activation.csv      # Mart: Activated trials

```









