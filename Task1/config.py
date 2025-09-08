"""
Configuration and constants for Trial Activation Analysis
"""

import matplotlib.pyplot as plt
import numpy as np

# Plotting configuration
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File paths
DATA_FILE = "data_clean.csv"
OUTPUT_DIR = "Task1"

# Analysis parameters
MIN_ACTIVITIES_THRESHOLD = 5
OUTLIER_THRESHOLD = 3  # Standard deviations for outlier detection
TRIAL_PERIOD_DAYS = 30

# Visualization settings
FIGURE_DPI = 300
PLOT_STYLE = 'seaborn-v0_8'
