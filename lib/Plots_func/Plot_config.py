"""
Configuration file for PPG plotting parameters
"""

import numpy as np
from pathlib import Path
import os 
import glob
import re

# ==========================================
# FILE PATHS
# ==========================================
# Base directory = folder where this script lives
base_dir = Path(__file__).parent

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "Results"
FIGURES_PATH = PROJECT_ROOT / "Plots"
IND_FIGURES_PATH = FIGURES_PATH / "ind_plot"

# Input data file
TEMP_RESULTS_FILE = RESULTS_PATH / "processed_ppg_results_temp.csv"
FREQ_RESULTS_FILE = RESULTS_PATH / "processed_ppg_results_freq.csv"
EDA_RESULTS_FILE = RESULTS_PATH / "processed_ppg_results_gsr.csv"
OUTLIER = True

# ==========================================
# EXPERIMENTAL STRUCTURE
# ==========================================
TASKS = ['noise', 'arith']
TASK_LABELS = {
    'Trial00': 'Baseline',
    'Trial01': 'Irrigation 1',
    'Trial02': 'Irrigation 2'}

PHASE_LABELS = {
    'baseline': 'Baseline',
    'anticipation': 'Anticipation', 
    'task': 'Task',
    'recovery': 'Recovery'
}

GROUPS = ['HC', 'CT', 'AT']
GROUP_LABELS = {
    'HC': 'Healthy Control',
    'CT': 'Chronic Tinnitus',
    'AT': 'Acute Tinnitus'
}

GROUP_COLORS = {
    'HC': 'tab:green',
    'CT': 'tab:orange', 
    'AT': 'tab:red'
}

# GROUP_COLORS = {
#     'HC': "#E89005", #'#B3D89C', 
#     'CT': 'darkblue',
#     'AT': '#A2C6FA'
# }


# ==========================================
# CONFIGURATION METRIC
# ==========================================


# === FIGURE SETTINGS ===
MULTIPANEL_FIGURE_WIDTH = 12
MULTIPANEL_ROW_HEIGHT = 3.5  # Height per metric row

# Line styling
PLOT_LINE_WIDTH = 2
PLOT_LINE_COLOR = 'tab:blue'
PLOT_CI_ALPHA = 0.3

# Y-axis settings
SHARE_Y_AXIS_ACROSS_TASKS = False  # Set to False for independent scaling


# ==========================================
# X-AXIS PARAMETERS
# ==========================================
BASELINE = [-90]               
BASELINE_VIRTUAL = -75            
BASELINE_X = -225          

ANTICIPATION = [-60, -30]
ANTICIPATION_START = -45

TASK = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
TASK_START = 15           
TASK_END = 285         

RECOVERY = [300, 330]
RECOVERY_END = 345   

EXCLUDE_BASELINE_FROM_PLOTS = True
if EXCLUDE_BASELINE_FROM_PLOTS == False: 
    EXPECTED_TIME = BASELINE + ANTICIPATION + TASK + RECOVERY
    EXPECTED_PLOT_TIME = [-90] + [t + 15 for t in ANTICIPATION + TASK + RECOVERY]
else: 
    EXPECTED_TIME = ANTICIPATION + TASK + RECOVERY
    EXPECTED_PLOT_TIME = [t + 15 for t in ANTICIPATION + TASK + RECOVERY]

# X-axis tick interval
X_TICK_INTERVAL = 30
X_AXIS_MIN = ANTICIPATION_START - X_TICK_INTERVAL    # -90
X_AXIS_MAX = RECOVERY_END + X_TICK_INTERVAL          # 360

EVENT_TIMES = [ANTICIPATION_START, TASK_START, TASK_END, RECOVERY_END]
EVENT_LABELS = ['Anticipation', 'Task Start', 'Task End', 'Recovery End']
EVENT_COLORS = ['black', 'black', 'black', 'black']


# ==========================================
# PLOTTING PARAMETERS
# ==========================================
# Figure settings
FIGURE_SIZE = (12, 4)  # Width, height for single row (2 tasks)
FONT_SIZE = 8
TITLE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 10

# Line and error bar styling
LINE_WIDTH = 2
ERROR_ALPHA = 0.3  # Transparency for confidence interval shading
CI_LEVEL = 0.95  # Confidence interval level (95%)

# Event line styling  
EVENT_LINE_STYLE = '--'
EVENT_LINE_WIDTH = 1.5
EVENT_LINE_ALPHA = 0.7

# Grid
SHOW_GRID = True
GRID_ALPHA = 0.3


# ==========================================
# Y-AXIS SCALING PARAMETERS
# ==========================================
# Y-axis scaling mode: 'global', 'per_row', or 'independent'
#   'global': Same y-axis range for all plots of the same metric (original behavior)
#   'per_row': Same y-axis range within each row (metric), but different across metrics
#   'independent': Each subplot gets its own tight y-axis range (best for seeing variations)
Y_AXIS_SCALING_MODE = 'per_row'
Y_AXIS_PADDING_FACTOR = 0.1 # 10% padding on extremities
Y_AXIS_NUM_TICKS = 6

# Global y-axis ranges per metric (populated at runtime by compute_global_metric_ranges)
METRIC_Y_RANGES = {}

# ==========================================
# STATISTICAL PARAMETERS
# ==========================================
CI_METHOD = 'ci'  # 'sem' for standard error, 'ci' for t-distribution CI
ALPHA_LEVEL = 0.05

# ==========================================
# OUTPUT SETTINGS
# ==========================================
# Save formats
SAVE_FORMATS = ['svg'] #['png', 'svg', 'pdf']
DPI = 30
PLOT_PREFIX = 'PPG'  # Prefix for saved plot files

# ==========================================
# VALUE COLUMN CONFIGURATION
# ==========================================
# Choose which column to plot:
#   'raw'                  - Raw metric values
#   'pct_change' - Percent change from baseline
#   'diff'       - Absolute difference from baseline
#   'log_ratio'  - Log ratio difference from baseline

VALUE_COLUMN = 'pct_change'

VALUE_COLUMN_LABELS = {
    'raw': '',                        
    'pct_change': '% Change from Baseline',
    'diff': 'Difference from Baseline',
    'log_ratio': 'Logarithmic difference from Baseline'
}

VALUE_COLUMN_FILENAME_SUFFIX = {
    'raw': '_raw',
    'pct_change': '_pctchange',
    'diff': '_diff'
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_task_moment_name(task, phase):
    """Construct task_moment name as it appears in CSV"""
    return f"{task}_{phase}"

def get_output_filename(metric, task=None, group=None, suffix=''):
    """Generate standardized output filename with value column indicator"""
    parts = [PLOT_PREFIX, metric]
    if task:
        parts.append(task)
    if group:
        parts.append(group)
    if suffix:
        parts.append(suffix.lstrip('_'))
    
    # Add value column suffix
    value_suffix = VALUE_COLUMN_FILENAME_SUFFIX.get(VALUE_COLUMN, '')
    parts.append(value_suffix.lstrip('_'))
    
    return '_'.join(parts)


