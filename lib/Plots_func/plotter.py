"""
Time-series plotting for interval metrics.

`plot_all_task_all_group` draws one figure per condition, overlaying the study
groups (HC, T) as lines of a metric's mean value across participants versus
within-trial relative time, with a shaded error band (±1 SEM by default, or a
t-distribution confidence interval).

All data handling -- interval extraction, aggregation and the task/recovery
event structure -- lives in `plot_utils` so it can be reused by other figures;
this module only turns those results into matplotlib figures.
"""

import matplotlib.pyplot as plt

from lib.Plots_func.plot_config import (
    FIGURES_PATH, SAVE_FORMATS, DPI, FIGURE_SIZE,
    FONT_SIZE, TITLE_FONT_SIZE, AXIS_LABEL_FONT_SIZE,
    LINE_WIDTH, ERROR_ALPHA, CI_LEVEL,
    SHOW_GRID, GRID_ALPHA,
    EVENT_LINE_STYLE, EVENT_LINE_WIDTH, EVENT_LINE_ALPHA,
    GROUP_COLORS, METRIC_CONFIG, VALUE_TYPE_LABELS,
)
from lib.Plots_func.plot_utils import (
    COL_GROUP, COL_PART, COL_COND, COL_REL_START, COL_REL_END,
    filter_interval_data, compute_mean_and_band, derive_event_segments,
    _draw_event_markers, _y_label, _band_label, _save,
)


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================
def plot_all_task_all_group(df, metric, value_type,
                            error_band='sem', ci_level=CI_LEVEL,
                            conditions=None, groups=None,
                            save=True, show=False,
                            figures_path=FIGURES_PATH, file_tag=''):
    """Plot mean interval metric vs within-trial time, one figure per condition.

    Parameters
    ----------
    df : DataFrame  -- a loaded results CSV (long format).
    metric : str    -- value of the `Metric` column to plot.
    value_type : str-- value of the `Value_type` column ('raw', 'diff',
                       'pct_change', 'log_ratio').
    error_band : str-- shaded band around the mean: 'sem' (default, ±1 SEM) or
                       'ci' (t-distribution confidence interval at ``ci_level``).
    ci_level : float-- confidence level used when ``error_band == 'ci'``.
    conditions : list, optional -- conditions to plot (default: every non-baseline
                       condition present, in first-seen order).
    groups : list, optional     -- groups to overlay (default: all present, sorted).
    save / show : bool
    figures_path : Path -- output root (subfolders per format are created).
    file_tag : str      -- optional suffix appended to output filenames.

    Returns a list of (condition, Figure) tuples.
    """
    d = filter_interval_data(df, metric, value_type)
    if d.empty:
        print(f"[plot_all_task_all_group] No interval rows for "
              f"metric={metric!r}, value_type={value_type!r}.")
        return []

    if conditions is None:
        conditions = [c for c in d[COL_COND].drop_duplicates()
                      if str(c) != 'baseline']
    if groups is None:
        groups = sorted(d[COL_GROUP].dropna().unique())

    band_lbl = _band_label(error_band, ci_level)

    results = []
    for condition in conditions:
        dc = d[d[COL_COND] == condition]
        if dc.empty:
            print(f"  (skip) condition {condition!r}: no data")
            continue

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        plotted_any = False
        for g in groups:
            dg = dc[dc[COL_GROUP] == g]
            if dg.empty:
                continue
            agg = compute_mean_and_band(dg, error_band=error_band, ci_level=ci_level)
            color = GROUP_COLORS.get(g)
            n_part = dg[COL_PART].nunique()
            ax.plot(agg['x'], agg['mean'], marker='o', markersize=4,
                    linewidth=LINE_WIDTH, color=color,
                    label=f"{g} (n={n_part})")
            band = agg.dropna(subset=['lower', 'upper'])
            if not band.empty:
                ax.fill_between(band['x'], band['lower'], band['upper'],
                                color=color, alpha=ERROR_ALPHA, linewidth=0)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        # Data-driven event markers (must run after lines set the y-limits).
        _draw_event_markers(ax, derive_event_segments(dc))

        # Data-driven x ticks: every interval start plus the final interval end.
        xticks = sorted(dc[COL_REL_START].unique())
        xticks.append(float(dc[COL_REL_END].max()))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=FONT_SIZE)

        ax.set_xlabel('Within-trial time (s) — interval start',
                      fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(_y_label(metric, value_type), fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_title(f"{metric} — {condition}  (band: {band_lbl})",
                     fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=FONT_SIZE)
        if SHOW_GRID:
            ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(fontsize=FONT_SIZE, loc='best')
        fig.tight_layout()

        if save:
            fname = f"timeseries_{metric}_{value_type}_{error_band}_{condition}{file_tag}"
            _save(fig, fname, figures_path)
        results.append((condition, fig))

    if show:
        plt.show()
    else:
        for _, fig in results:
            plt.close(fig)

    return results
