
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
TF_RESULTS_FILE = RESULTS_PATH / "processed_ppg_results_freq.csv"
GSR_RESULTS_FILE = RESULTS_PATH / "processed_gsr_results.csv"
VAS_RESULTS_FILE = RESULTS_PATH / "processed_vas_results.csv"
OUTLIER = True


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
EXCLUDE_BASELINE_FROM_PLOTS = True

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
    'diff': 'Difference from Baseline'
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

# ==========================================
# TIME-SERIES PLOT STYLING (data-driven plots)
# ==========================================
# Colours for the study groups overlaid as lines (one figure per condition).
# HC = SBSA healthy controls, T = SBAA tinnitus. Unknown groups fall back to
# matplotlib's default colour cycle.
GROUP_COLORS = {
    'HC': 'tab:blue',
    'T':  'tab:red',
}

# Y-axis labels per metric. Keys must match the `Metric` column in the CSVs.
METRIC_CONFIG = {
    # --- temporal HRV ---
    'RMSSD':    {'label': 'RMSSD (ms)'},
    'SDNN':     {'label': 'SDNN (ms)'},
    'mean_HR':  {'label': 'Mean HR (bpm)'},
    'mean_RRI': {'label': 'Mean RRI (ms)'},
    # --- frequency HRV ---
    'VLF': {'label': 'VLF power (ms²)'},
    'LF':  {'label': 'LF power (ms²)'},
    'HF':  {'label': 'HF power (ms²)'},
    # --- GSR / EDA ---
    'Tonic_SCL_mean':             {'label': 'Tonic SCL mean (µS)'},
    'Tonic_SCL_slope':            {'label': 'Tonic SCL slope (µS/s)'},
    'Phasic_AUC':                 {'label': 'Phasic AUC (µS·s)'},
    'Phasic_SCR_amplitude_mean':  {'label': 'SCR amplitude mean (µS)'},
    'Phasic_SCR_amplitude_sum':   {'label': 'SCR amplitude sum (µS)'},
    'Phasic_SCR_count':           {'label': 'SCR count'},
    'Phasic_SCR_rate':            {'label': 'SCR rate (per min)'},
    # --- subjective VAS ---
    'VAS_mean':   {'label': 'VAS mean'},
    'VAS_median': {'label': 'VAS median'},
    'VAS_std':    {'label': 'VAS std'},
}

# Axis-label suffix per Value_type (the `Value_type` column selects which is
# plotted; the numeric always comes from the `Value` column).
VALUE_TYPE_LABELS = {
    'raw':        '',
    'diff':       'Δ from baseline',
    'pct_change': '% change from baseline',
    'log_ratio':  'log-ratio vs baseline',
}


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


