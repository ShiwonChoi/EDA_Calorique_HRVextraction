"""
Reusable data utilities for interval-metric plotting.

These helpers are schema-driven and metric-agnostic, so they work unchanged on
every results file that follows the shared long-format layout (temporal HRV,
frequency HRV, GSR/EDA, VAS, ...). They cover the three steps common to any such
plot:

  1. `filter_interval_data`  -- pull the interval rows for one metric/value_type.
  2. `compute_mean_and_band` -- mean across participants with a selectable band.
  3. `derive_event_segments` -- reconstruct the task/recovery event structure
                                from the data itself (no hardcoded boundaries).

Nothing here imports matplotlib; keep this module plotting-free so it can be
reused for tables, stats, or other figure types.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from lib.Plots_func.plot_config import (
    SAVE_FORMATS, DPI, FONT_SIZE,
    EVENT_LINE_STYLE, EVENT_LINE_WIDTH, EVENT_LINE_ALPHA,
    METRIC_CONFIG, VALUE_TYPE_LABELS,
)

# ==========================================================================
# SCHEMA -- column names shared by every results CSV. A schema change is a
# one-line edit here and every downstream plot follows.
# ==========================================================================
COL_GROUP     = 'groupe'
COL_PART      = 'participant'
COL_COND      = 'condition'
COL_MOMENT    = 'task_moment'
COL_RECTYPE   = 'recording_type'
COL_METRIC    = 'Metric'
COL_VTYPE     = 'Value_type'
COL_VALUE     = 'Value'
COL_REL_START = 'time_interval_rel_start'
COL_REL_END   = 'time_interval_rel_end'

# Supported error-band methods for the shaded region around the mean.
ERROR_BANDS = ('sem', 'ci')


# ==========================================================================
# EXTRACTION
# ==========================================================================
def filter_interval_data(df, metric, value_type, require_success=True,
                         round_decimals=0):
    """Return the interval rows for a single metric and value_type, cleaned.

    Keeps only ``recording_type == 'interval'`` rows (dropping per-trial `total`
    summaries and the interval-less `baseline` trial), coerces the numeric
    columns, and drops rows with no value or no interval start.

    The interval start (``time_interval_rel_start``) is rounded so that values
    that should coincide across trials/participants but differ only by
    floating-point noise (e.g. 299.999 vs 300.0) collapse onto the same x. This
    is what makes them aggregate into one point instead of two. ``round_decimals``
    controls the precision; ``0`` (default) snaps to the nearest second, matching
    the 30 s bin grid used across every results file.

    Parameters
    ----------
    df : DataFrame       -- a loaded results CSV (long format).
    metric : str         -- value of the `Metric` column.
    value_type : str     -- value of the `Value_type` column ('raw', 'diff',
                            'pct_change', 'log_ratio').
    require_success : bool-- if True and a `status` column exists, keep only
                            rows with status == 'SUCCESS'.
    round_decimals : int  -- decimal places to round the interval start/end to
                            before grouping. Use ``None`` to disable rounding.

    Returns a filtered copy (may be empty).
    """
    d = df[(df[COL_RECTYPE] == 'interval') &
           (df[COL_METRIC] == metric) &
           (df[COL_VTYPE] == value_type)].copy()

    if require_success and 'status' in d.columns:
        d = d[d['status'] == 'SUCCESS']

    d[COL_VALUE] = pd.to_numeric(d[COL_VALUE], errors='coerce')
    for c in (COL_REL_START, COL_REL_END):
        d[c] = pd.to_numeric(d[c], errors='coerce')
        if round_decimals is not None:
            d[c] = d[c].round(round_decimals)

    return d.dropna(subset=[COL_VALUE, COL_REL_START])


# ==========================================================================
# AGGREGATION
# ==========================================================================
def compute_mean_and_band(d, error_band='sem', ci_level=0.95):
    """Mean of Value across participants, with a selectable error band.

    Adapted from ``Load_data.compute_mean_and_ci`` for this study's schema: the
    x-axis key is ``time_interval_rel_start`` (the interval START, so the point
    at x=0 summarises [0, first_end)).

    Parameters
    ----------
    d : DataFrame     -- interval rows for a single (group, condition).
    error_band : str  -- 'sem' (default) shades mean ± 1 standard error of the
                         mean; 'ci' shades a two-sided t-distribution confidence
                         interval at ``ci_level`` (the original behaviour).
    ci_level : float  -- confidence level used when ``error_band == 'ci'``.

    Returns a DataFrame sorted by ``x`` with columns: x, mean, lower, upper, n.
    ``lower``/``upper`` are NaN wherever a bin has fewer than two participants
    (a band from a single participant is undefined).
    """
    if error_band not in ERROR_BANDS:
        raise ValueError(f"error_band must be one of {ERROR_BANDS}, got {error_band!r}")

    grp = d.groupby(COL_REL_START)[COL_VALUE]
    agg = grp.agg(mean='mean', std='std', n='count').reset_index()
    agg = agg.sort_values(COL_REL_START).rename(columns={COL_REL_START: 'x'})

    n    = agg['n'].to_numpy()
    std  = agg['std'].to_numpy()
    mean = agg['mean'].to_numpy()

    margin = np.full(mean.shape, np.nan, dtype=float)
    ok = n >= 2
    if ok.any():
        sem = std[ok] / np.sqrt(n[ok])
        if error_band == 'sem':
            margin[ok] = sem
        else:  # 'ci' -- two-sided t interval
            t_crit = scipy_stats.t.ppf(0.5 + ci_level / 2.0, n[ok] - 1)
            margin[ok] = t_crit * sem

    agg['lower'] = mean - margin
    agg['upper'] = mean + margin
    return agg[['x', 'mean', 'lower', 'upper', 'n']]


# ==========================================================================
# EVENT STRUCTURE (derived from the data)
# ==========================================================================
def derive_event_segments(dc):
    """Collapse the interval rows of one condition into ordered task_moment runs.

    Boundaries and widths are read from the actual ``time_interval_rel_start`` /
    ``time_interval_rel_end`` values, so no interval width or task/recovery
    boundary is assumed. Contiguous intervals sharing a ``task_moment`` merge
    into one segment.

    Returns a list of dicts ``{start, end, task_moment, label}`` in time order.
    ``label`` is the condition name for ``task`` segments, otherwise the
    ``task_moment`` value (e.g. 'recovery').
    """
    intervals = (
        dc.groupby(COL_REL_START)
          .agg(end=(COL_REL_END, 'max'),
               task_moment=(COL_MOMENT, 'first'),
               condition=(COL_COND, 'first'))
          .reset_index()
          .sort_values(COL_REL_START)
    )

    segments = []
    for _, row in intervals.iterrows():
        tm    = row['task_moment']
        start = row[COL_REL_START]
        end   = row['end']
        if segments and segments[-1]['task_moment'] == tm:
            segments[-1]['end'] = max(segments[-1]['end'], end)
        else:
            label = row['condition'] if tm == 'task' else tm
            segments.append({'start': start, 'end': end,
                             'task_moment': tm, 'label': label})
    return segments


# ==========================================================================
# PLOTTING HELPERS
# ==========================================================================
def _draw_event_markers(ax, segments):
    """Draw boundary lines between task_moment segments and label each span."""
    if not segments:
        return
    y0, y1 = ax.get_ylim()
    y_text = y1 - (y1 - y0) * 0.03

    boundaries = [segments[0]['start']] + [s['end'] for s in segments]
    for xb in boundaries:
        ax.axvline(xb, color='0.45', linestyle=EVENT_LINE_STYLE,
                   linewidth=EVENT_LINE_WIDTH, alpha=EVENT_LINE_ALPHA)

    for seg in segments:
        x_center = (seg['start'] + seg['end']) / 2.0
        ax.text(x_center, y_text, str(seg['label']),
                ha='center', va='top', rotation=0, fontsize=FONT_SIZE,
                color='0.25',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='0.7', alpha=0.75))


def _y_label(metric, value_type):
    base = METRIC_CONFIG.get(metric, {}).get('label', metric)
    suffix = VALUE_TYPE_LABELS.get(value_type, value_type)
    return base if not suffix else f"{base} — {suffix}"


def _band_label(error_band, ci_level):
    return "±1 SEM" if error_band == 'sem' else f"{int(round(ci_level * 100))}% CI"


def _save(fig, filename, figures_path):
    figures_path.mkdir(parents=True, exist_ok=True)
    for fmt in SAVE_FORMATS:
        out_dir = figures_path / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved: {path}")
