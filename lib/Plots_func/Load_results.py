"""
PPG Metric Plotting Script
Functions for plotting time-series data with confidence intervals and event markers

UPDATED: 
- Added flexible y-axis scaling modes: 'global', 'per_row', or 'independent'
- Improved tight y-axis calculation for better visualization of variations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Import configuration
from lib.Plotting.plot_config import *

# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================

def load_data(filepath=PPG_RESULTS_FILE):
    print("="*70)
    print("Loading PPG Data")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    print(
        f"Successful rows: {(df.status == 'SUCCESS').sum()} / "
        f"Failed rows: {(df.status == 'FAILED').sum()}")

    # Filter for successful processing only
    df_success = df[df["status"] == "SUCCESS"].copy()
    
    print(f"Unique participants: {df_success['participant_id'].nunique()}")
    print(f"Available metrics: {df_success['Metric'].unique().tolist()}")
    print(f"Groups present: {df_success['groupe'].unique().tolist()}")
    print()
    
    # Build participant-count dictionary
    participant_counts = {}
    
    print("Participants per group:")
    for grp in df_success['groupe'].unique():
        n = df_success[df_success['groupe'] == grp]['participant_id'].nunique()
        participant_counts[grp] = n
        print(f"  {grp}: {n} participants")
    print()
    
    return df_success, participant_counts


def extract_task_data(df, task, metric, vtype=VALUE_COLUMN, exclude_baseline=EXCLUDE_BASELINE_FROM_PLOTS, group=None):
    # Filter for this task type
    task_mask = df['task'].str.contains(task)
    metric_mask = df['Metric'] == metric
    value_mask = df["Value_type"] == vtype

    # Combine filters
    filtered = df[task_mask & metric_mask & value_mask].copy()
    
    # Filter by group if specified
    if group is not None:
        filtered = filtered[filtered['groupe'] == group]
    
    # Exclude baseline if requested
    if exclude_baseline:
        filtered = filtered[~filtered['task_moment'].str.contains('baseline')]
    
    return filtered


def compute_mean_and_ci(df, ci_method=CI_METHOD, ci_level=CI_LEVEL):
    """
    Compute mean and confidence intervals over participants
    
    Returns:
        x_time: Time points
        mean_values: Mean across participants
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
    """
    if EXCLUDE_BASELINE_FROM_PLOTS == False:
        df_filtered = df.copy()
    else:
        df_filtered = df[~df["task_moment"].str.contains("baseline", case=False)]

    # Group by time and compute statistics
    grouped = df_filtered.groupby("time_interval_relative")['Value']

    x_time = np.array(sorted(df_filtered["time_interval_relative"].unique()))
    x_plot = np.array(sorted(df_filtered["time_center_plot"].unique()))
    mean_values = grouped.mean().reindex(x_time).values
    
    if ci_method == 'sem':
        # Standard error of the mean
        sem = grouped.sem().reindex(x_time).values
        ci_lower = mean_values - sem
        ci_upper = mean_values + sem
        
    elif ci_method == 'ci':
        # t-distribution confidence interval
        std = grouped.std().reindex(x_time).values
        n = grouped.count().reindex(x_time).values
        
        # t-critical value
        alpha = 1 - ci_level
        df_freedom = np.maximum(n - 1, 1)  # Ensure at least 1 degree of freedom
        t_crit = stats.t.ppf(1 - alpha/2, df_freedom)
        
        margin = t_crit * (std / np.sqrt(np.maximum(n, 1)))
        ci_lower = mean_values - margin
        ci_upper = mean_values + margin
    
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")
    
    return x_time, x_plot, mean_values, ci_lower, ci_upper


def filter_outlier_bins(df, n_std=3, verbose=True, save_report=OUTLIER, report_name='outlier_report'):
    """

    Returns:
        df_clean    : DataFrame with outlier rows removed
        flag_report : dict keyed by (time_interval_relative, Metric)
    """
    df = df.copy()
    keep_mask = pd.Series(True, index=df.index)
    flag_report = {}

    for (t_bin, metric), grp in df.groupby(['time_interval_relative', 'Metric']):
        values = grp['Value']
        #bin_mean = values.mean()
        bin_median = values.median()
        bin_std  = values.std()

        if bin_std == 0 or np.isnan(bin_std):
            flag_report[(t_bin, metric)] = {
                'n_total': len(grp), 'n_flagged': 0,
                'flagged_participants': [], 'bin_mean': bin_median, 'bin_std': bin_std
            }
            continue

        outlier_mask = (values - bin_median).abs() > n_std * bin_std
        flagged_pids = grp.loc[outlier_mask, 'participant_id'].tolist()
        keep_mask.loc[grp.index] = ~outlier_mask

        flag_report[(t_bin, metric)] = {
            'n_total':              len(grp),
            'n_flagged':            outlier_mask.sum(),
            'flagged_participants': flagged_pids,
            'bin_mean':             bin_median,
            'bin_std':              bin_std,
        }

    df_clean = df[keep_mask]
    total_flagged = sum(v['n_flagged'] for v in flag_report.values())

    # ---- Console report ----
    if verbose:
        print(f"{'Bin':>6}  {'Metric':<12}  {'Total':>5}  {'Flagged':>7}  Participants removed")
        print("-" * 65)
        for (t_bin, metric), info in sorted(flag_report.items()):
            if info['n_flagged'] > 0:
                pids = ', '.join(info['flagged_participants'])
                print(f"{t_bin:>6}  {metric:<12}  {info['n_total']:>5}  "
                      f"{info['n_flagged']:>7}  {pids}")
        print(f"\nTotal flagged: {total_flagged} / {len(df)} rows")

    # ---- Markdown report ----
    if save_report:
        import datetime
        lines = []
        lines.append(f"# Outlier Bin Report")
        lines.append(f"\n**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  \n**Threshold:** ±{n_std} SD per bin")
        lines.append(f"  \n**Total flagged:** {total_flagged} / {len(df)} rows")

        # Group by metric for readability
        metrics = sorted(set(m for _, m in flag_report.keys()))
        for metric in metrics:
            lines.append(f"\n## {metric}\n")
            lines.append(f"| Bin (s) | Kept | Flagged | Total | Flagged Participants |")
            lines.append(f"|--------:|-----:|--------:|------:|----------------------|")
            for (t_bin, m), info in sorted(flag_report.items()):
                if m != metric:
                    continue
                kept = info['n_total'] - info['n_flagged']
                pids = ', '.join(info['flagged_participants']) if info['flagged_participants'] else '—'
                lines.append(
                    f"| {t_bin:>7} | {kept:>4} | {info['n_flagged']:>7} "
                    f"| {info['n_total']:>5} | {pids} |"
                )

        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        report_path = FIGURES_PATH / f"{report_name}.md"
        report_path.write_text('\n'.join(lines))
        print(f"\nReport saved: {report_path}")

    return df_clean, flag_report


def compute_global_metric_ranges(df, metrics=None):
    """
    Compute global y-axis ranges for each metric across all tasks and groups.
    Only used when Y_AXIS_SCALING_MODE = 'global'
    """
    global METRIC_Y_RANGES
    
    if Y_AXIS_SCALING_MODE != 'global':
        print(f"Y-axis scaling mode is '{Y_AXIS_SCALING_MODE}', skipping global range computation")
        return {}
    
    if metrics is None:
        metrics = df['Metric'].unique()
    
    print("="*70)
    print("Computing Global Y-Axis Ranges")
    print("="*70)
    
    for metric in metrics:
        all_values = []
        
        # Collect values across all tasks and groups
        for task in TASKS:
            for group in GROUPS:
                task_data = extract_task_data(df, task, metric, 
                                              exclude_baseline=EXCLUDE_BASELINE_FROM_PLOTS, 
                                              group=group)
                if len(task_data) > 0:
                    values = task_data[VALUE_COLUMN].dropna().values
                    all_values.extend(values)
        
        if len(all_values) > 0:
            y_min, y_max = compute_tight_ylim(all_values)
            METRIC_Y_RANGES[metric] = (y_min, y_max)
            print(f"  {metric}: {y_min:.2f} to {y_max:.2f}")
        else:
            METRIC_Y_RANGES[metric] = None
            print(f"  {metric}: No data")
    
    print()
    return METRIC_Y_RANGES


# ==========================================
# PLOTTING HELPER FUNCTIONS
# ==========================================

def plot_event_lines(ax, show_labels=True):
    ylim = ax.get_ylim()
    
    for time, label, color in zip(EVENT_TIMES, EVENT_LABELS, EVENT_COLORS):
        ax.axvline(
            x=time,
            color=color,
            linestyle=EVENT_LINE_STYLE,
            linewidth=EVENT_LINE_WIDTH,
            alpha=EVENT_LINE_ALPHA
        )
        
        if show_labels:
            ax.text(
                time + 2,  # Slight offset to avoid overlap with line
                ylim[1] - (ylim[1] - ylim[0]) * 0.05,  # Slightly below top
                label,
                rotation=90, 
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=FONT_SIZE-2,
                color=color,
                alpha=0.8
            )


def remap_baseline_x_values(x_array):
    """
    Remap baseline x-values from their true position (-240) to a virtual
    display position one X_TICK_INTERVAL before the first non-baseline tick.

    This keeps the baseline point visually adjacent to the rest of the
    timeline (at -90) without the large spatial gap that -240 would create.
    Call this on x_time BEFORE passing it to ax.plot / ax.fill_between.

    Args:
        x_array: array-like of x (time_interval_relative) values

    Returns:
        np.ndarray: copy of x_array with -240 replaced by BASELINE_VIRTUAL_X
    """
    x = np.array(x_array, dtype=float)
    x[x <= BASELINE_X] = BASELINE_VIRTUAL
    return x


def get_y_ticks(y_min, y_max, num_ticks=None):
    """
    Generate y-axis tick positions and labels with nice round intervals
    
    Args:
        y_min: Minimum y value
        y_max: Maximum y value
        num_ticks: Approximate number of ticks (default: Y_AXIS_NUM_TICKS from config)
    
    Returns:
        ticks: Array of tick positions
        labels: List of tick labels as strings
    """
    if num_ticks is None:
        num_ticks = Y_AXIS_NUM_TICKS
    
    # Calculate the range
    data_range = y_max - y_min
    
    # Determine nice interval based on range
    # Target interval that gives approximately num_ticks ticks
    raw_interval = data_range / (num_ticks - 1)
    
    # Round to nice numbers (1, 2, 5, 10, 20, 50, 100, etc.)
    magnitude = 10 ** np.floor(np.log10(raw_interval))
    normalized = raw_interval / magnitude
    
    # Choose nice interval
    if normalized < 1.5:
        nice_interval = 1 * magnitude
    elif normalized < 3:
        nice_interval = 2 * magnitude
    elif normalized < 7:
        nice_interval = 5 * magnitude
    else:
        nice_interval = 10 * magnitude
    
    # Handle very small intervals (< 1)
    if nice_interval < 1:
        if nice_interval < 0.1:
            nice_interval = 0.1
        elif nice_interval < 0.2:
            nice_interval = 0.2
        elif nice_interval < 0.5:
            nice_interval = 0.5
        else:
            nice_interval = 1.0
    
    # Generate ticks starting from a nice round number
    tick_min = np.floor(y_min / nice_interval) * nice_interval
    tick_max = np.ceil(y_max / nice_interval) * nice_interval
    
    ticks = np.arange(tick_min, tick_max + nice_interval/2, nice_interval)
    
    # Format labels based on the interval size
    if nice_interval >= 1:
        # Integer labels for intervals >= 1
        labels = [f"{int(t)}" if t == int(t) else f"{t:.1f}" for t in ticks]
    elif nice_interval >= 0.1:
        # One decimal place for intervals >= 0.1
        labels = [f"{t:.1f}" for t in ticks]
    else:
        # Two decimal places for smaller intervals
        labels = [f"{t:.2f}" for t in ticks]
    
    return ticks, labels


def compute_tight_ylim(values, padding_factor=Y_AXIS_PADDING_FACTOR):
    """
    Compute tight y-axis limits based on data
    
    Args:
        values: Array of values to compute limits for
        padding_factor: Fraction of range to add as padding
    
    Returns:
        (y_min, y_max): Tuple of y-axis limits
    """
    if len(values) == 0:
        return (0, 1)
    
    y_min = np.nanmin(values)
    y_max = np.nanmax(values)
    y_range = y_max - y_min
    
    # Handle case where all values are the same
    if y_range == 0:
        y_range = abs(y_min) * 0.1 if y_min != 0 else 1
    
    padding = y_range * padding_factor
    
    return (y_min - padding, y_max + padding)


def get_y_label(metric):
    """
    Get appropriate y-axis label based on VALUE_COLUMN setting
    
    Args:
        metric: metric name (e.g., 'RMSSD')
    
    Returns:
        str: y-axis label
    """
    if VALUE_COLUMN == 'Value':
        # Use metric's configured label
        config = METRIC_CONFIG.get(metric, {})
        return config.get('label', metric)
    else:
        # Use the generic label for percent/diff change
        base_label = VALUE_COLUMN_LABELS.get(VALUE_COLUMN, VALUE_COLUMN)
        return f"{metric} - {base_label}"


def format_axes(ax, x_time, x_plot, ylabel=None, show_xlabel=True, metric=None, ylim=None,
                include_baseline=False):
    
    # ==========================================
    # X-AXIS FORMATTING
    # ==========================================

    # Validate
    if list(x_time) != list(EXPECTED_TIME):
        print(f"WARNING: x_time mismatch\n  got:      {x_time}\n  expected: {EXPECTED_TIME}")
    if list(x_plot) != list(EXPECTED_PLOT_TIME):
        print(f"WARNING: x_plot mismatch\n  got:      {x_plot}\n  expected: {EXPECTED_PLOT_TIME}")

    # X-axis: ticks at bin boundaries with padding
    x_ticks = np.append(x_time, x_time[-1] + X_TICK_INTERVAL)
    ax.set_xticks(x_ticks)
    ax.set_xlim(x_ticks[0] - X_TICK_INTERVAL, x_ticks[-1] + X_TICK_INTERVAL)
    

    # ==========================================
    # Y-AXIS FORMATTING
    # ==========================================

    # Y-axis formatting based on scaling mode
    if ylim is not None:
        # Use provided ylim (highest priority)
        ax.set_ylim(ylim)
    elif Y_AXIS_SCALING_MODE == 'global' and metric and metric in METRIC_Y_RANGES and METRIC_Y_RANGES[metric]:
        # Use global range for this metric
        ax.set_ylim(METRIC_Y_RANGES[metric])
    
    # Set y-ticks with numeric labels
    ylim_current = ax.get_ylim()
    y_ticks, y_labels = get_y_ticks(ylim_current[0], ylim_current[1])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE)
    
    if show_xlabel:
        ax.set_xlabel('Time (s)', fontsize=AXIS_LABEL_FONT_SIZE)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    
    # Grid
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)


# ==========================================
# SAVE FIGURE
# ==========================================

def _save_figure(fig, filename, figures_path=None):
    base_path = figures_path if figures_path is not None else FIGURES_PATH
    base_path.mkdir(parents=True, exist_ok=True)
    for fmt in SAVE_FORMATS:
        fmt_path = base_path / fmt
        fmt_path.mkdir(parents=True, exist_ok=True)
        filepath = fmt_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"  Saved: {filepath}")