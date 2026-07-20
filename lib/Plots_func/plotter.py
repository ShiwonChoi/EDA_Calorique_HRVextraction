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

import colorsys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from lib.Plots_func.plot_config import (
    FIGURES_PATH, SAVE_FORMATS, DPI, FIGURE_SIZE,
    FONT_SIZE, TITLE_FONT_SIZE, AXIS_LABEL_FONT_SIZE,
    LINE_WIDTH, ERROR_ALPHA, CI_LEVEL,
    SHOW_GRID, GRID_ALPHA,
    EVENT_LINE_STYLE, EVENT_LINE_WIDTH, EVENT_LINE_ALPHA,
    GROUP_COLORS, METRIC_CONFIG, VALUE_TYPE_LABELS,
)
from lib.Plots_func.plot_utils import (
    COL_GROUP, COL_PART, COL_COND, COL_MOMENT, COL_REL_START, COL_REL_END,
    filter_interval_data, filter_baseline_data,
    compute_mean_and_band, derive_event_segments,
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


# ==========================================================================
# BASELINE vs RECOVERY COMPARISON
# ==========================================================================
def baseline_comparison(df, metric, value_type='raw',
                        error_band='sem', ci_level=CI_LEVEL,
                        groups=None, save=True, show=False,
                        figures_path=FIGURES_PATH, file_tag=''):
    """Compare the baseline reference against recovery, HC vs T, in one figure.

    Draws four elements sharing the recovery time axis:

      * recovery curve for each group -- the mean interval metric across the
        recovery bins (pooled over every task condition), with a shaded error
        band, in the same style as `plot_all_task_all_group`;
      * baseline reference for each group -- a single mean drawn as a flat
        (dashed) horizontal line spanning the recovery x-range, with its error
        band shaded across that span. The baseline trial has no interval
        breakdown (one ``total`` per participant), so it is one value, not a
        curve.

    Groups are distinguished by colour (``GROUP_COLORS``); task_moment by line
    style: recovery is solid with markers, baseline is dashed and flat.

    Parameters
    ----------
    df : DataFrame  -- a loaded results CSV (long format).
    metric : str    -- value of the `Metric` column to plot.
    value_type : str-- value of the `Value_type` column; defaults to 'raw'.
    error_band : str-- shaded band: 'sem' (default, +/-1 SEM) or 'ci'.
    ci_level : float-- confidence level used when ``error_band == 'ci'``.
    groups : list, optional -- groups to overlay (default: all present, sorted).
    save / show : bool
    figures_path : Path -- output root (subfolders per format are created).
    file_tag : str      -- optional suffix appended to the output filename.

    Returns the matplotlib Figure, or None if there is nothing to plot.
    """
    rec = filter_interval_data(df, metric, value_type)
    rec = rec[rec[COL_MOMENT] == 'recovery']
    base = filter_baseline_data(df, metric, value_type)

    if rec.empty and base.empty:
        print(f"[baseline_comparison] No baseline or recovery rows for "
              f"metric={metric!r}, value_type={value_type!r}.")
        return None

    if groups is None:
        present = pd.concat([rec[COL_GROUP], base[COL_GROUP]]).dropna()
        groups = sorted(present.unique())

    band_lbl = _band_label(error_band, ci_level)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Recovery x-extent -- also the span across which the flat baseline is drawn.
    if not rec.empty:
        x_min = float(rec[COL_REL_START].min())
        x_max = float(rec[COL_REL_END].max())
    else:
        x_min, x_max = 0.0, 1.0

    plotted_any = False
    for g in groups:
        color = GROUP_COLORS.get(g)

        # Recovery -- interval curve with error band.
        rg = rec[rec[COL_GROUP] == g]
        if not rg.empty:
            agg = compute_mean_and_band(rg, error_band=error_band, ci_level=ci_level)
            n_part = rg[COL_PART].nunique()
            ax.plot(agg['x'], agg['mean'], marker='o', markersize=4,
                    linewidth=LINE_WIDTH, color=color,
                    label=f"{g} recovery (n={n_part})")
            b = agg.dropna(subset=['lower', 'upper'])
            if not b.empty:
                ax.fill_between(b['x'], b['lower'], b['upper'],
                                color=color, alpha=ERROR_ALPHA, linewidth=0)
            plotted_any = True

        # Baseline -- single mean drawn as a flat dashed reference + band.
        bg = base[base[COL_GROUP] == g]
        if not bg.empty:
            bagg = compute_mean_and_band(bg, error_band=error_band, ci_level=ci_level)
            row = bagg.iloc[0]
            n_part = bg[COL_PART].nunique()
            ax.plot([x_min, x_max], [row['mean'], row['mean']],
                    linestyle='--', linewidth=LINE_WIDTH, color=color,
                    label=f"{g} baseline (n={n_part})")
            if pd.notna(row['lower']) and pd.notna(row['upper']):
                ax.fill_between([x_min, x_max], row['lower'], row['upper'],
                                color=color, alpha=ERROR_ALPHA, linewidth=0)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"[baseline_comparison] Nothing to plot for metric={metric!r}.")
        return None

    # Data-driven x ticks: every recovery interval start plus the final end.
    if not rec.empty:
        xticks = sorted(rec[COL_REL_START].unique())
        xticks.append(x_max)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=FONT_SIZE)

    ax.set_xlabel('Within-trial time (s) — interval start',
                  fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(_y_label(metric, value_type), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(f"{metric} — baseline vs recovery  (band: {band_lbl})",
                 fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_SIZE, loc='best')
    fig.tight_layout()

    if save:
        fname = f"baseline_comparison_{metric}_{value_type}_{error_band}{file_tag}"
        _save(fig, fname, figures_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ==========================================================================
# BASELINE vs PER-CONDITION RECOVERY COMPARISON
# ==========================================================================
def _hue_variants(base_color, n, l_range=(0.30, 0.72)):
    """Return ``n`` same-hue variants of ``base_color``, dark -> light.

    The base colour's hue (and, as far as possible, saturation) is preserved;
    only lightness is swept across ``l_range`` so the variants read as shades of
    one colour family -- e.g. several blues for HC, several reds for T. With
    ``n == 1`` the single variant sits at the middle of the range.
    """
    r, g, b = mcolors.to_rgb(base_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    if n == 1:
        lights = [sum(l_range) / 2.0]
    else:
        lo, hi = l_range
        lights = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [colorsys.hls_to_rgb(h, li, s) for li in lights]


def baseline_comparison_by_condition(df, metric, value_type='raw',
                                     error_band='sem', ci_level=CI_LEVEL,
                                     conditions=None, groups=None,
                                     show_bands=False, save=True, show=False,
                                     figures_path=FIGURES_PATH, file_tag=''):
    """Compare the baseline reference against each condition's recovery, HC vs T.

    Like `baseline_comparison`, but instead of pooling recovery over all task
    conditions it draws a separate recovery curve per condition. Each group's
    condition curves are drawn in different lightness variants of that group's
    base colour (``GROUP_COLORS``), so the whole HC family reads as blues and the
    whole T family as reds; the flat dashed baseline reference for each group is
    drawn in the pure group colour.

    Parameters
    ----------
    df : DataFrame  -- a loaded results CSV (long format).
    metric : str    -- value of the `Metric` column to plot.
    value_type : str-- value of the `Value_type` column; defaults to 'raw'.
    error_band : str-- shaded band: 'sem' (default, +/-1 SEM) or 'ci'.
    ci_level : float-- confidence level used when ``error_band == 'ci'``.
    conditions : list, optional -- recovery conditions to draw (default: every
                       recovery condition present, in first-seen order).
    groups : list, optional     -- groups to overlay (default: all present, sorted).
    show_bands : bool-- draw the error band around each curve. With many
                       conditions overlaid the bands can crowd the figure, so
                       they are drawn at a reduced alpha and can be turned off.
    save / show : bool
    figures_path : Path -- output root (subfolders per format are created).
    file_tag : str      -- optional suffix appended to the output filename.

    Returns the matplotlib Figure, or None if there is nothing to plot.
    """
    rec = filter_interval_data(df, metric, value_type)
    rec = rec[rec[COL_MOMENT] == 'recovery']
    base = filter_baseline_data(df, metric, value_type)

    if rec.empty and base.empty:
        print(f"[baseline_comparison_by_condition] No baseline or recovery rows "
              f"for metric={metric!r}, value_type={value_type!r}.")
        return None

    if conditions is None:
        conditions = list(rec[COL_COND].drop_duplicates())
    if groups is None:
        present = pd.concat([rec[COL_GROUP], base[COL_GROUP]]).dropna()
        groups = sorted(present.unique())

    band_lbl = _band_label(error_band, ci_level)
    band_alpha = ERROR_ALPHA * 0.4  # reduced -- many curves overlap here.

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if not rec.empty:
        x_min = float(rec[COL_REL_START].min())
        x_max = float(rec[COL_REL_END].max())
    else:
        x_min, x_max = 0.0, 1.0

    plotted_any = False
    for g in groups:
        base_color = GROUP_COLORS.get(g)
        shades = _hue_variants(base_color, len(conditions))

        # Per-condition recovery curves, each a distinct shade of the group hue.
        for cond, shade in zip(conditions, shades):
            rg = rec[(rec[COL_GROUP] == g) & (rec[COL_COND] == cond)]
            if rg.empty:
                continue
            agg = compute_mean_and_band(rg, error_band=error_band, ci_level=ci_level)
            n_part = rg[COL_PART].nunique()
            ax.plot(agg['x'], agg['mean'], marker='o', markersize=3,
                    linewidth=LINE_WIDTH, color=shade,
                    label=f"{g} {cond} (n={n_part})")
            if show_bands:
                b = agg.dropna(subset=['lower', 'upper'])
                if not b.empty:
                    ax.fill_between(b['x'], b['lower'], b['upper'],
                                    color=shade, alpha=band_alpha, linewidth=0)
            plotted_any = True

        # Baseline reference -- flat dashed line in the pure group colour.
        bg = base[base[COL_GROUP] == g]
        if not bg.empty:
            bagg = compute_mean_and_band(bg, error_band=error_band, ci_level=ci_level)
            row = bagg.iloc[0]
            n_part = bg[COL_PART].nunique()
            ax.plot([x_min, x_max], [row['mean'], row['mean']],
                    linestyle='--', linewidth=LINE_WIDTH + 0.5, color=base_color,
                    label=f"{g} baseline (n={n_part})")
            if show_bands and pd.notna(row['lower']) and pd.notna(row['upper']):
                ax.fill_between([x_min, x_max], row['lower'], row['upper'],
                                color=base_color, alpha=band_alpha, linewidth=0)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"[baseline_comparison_by_condition] Nothing to plot for "
              f"metric={metric!r}.")
        return None

    # Data-driven x ticks: every recovery interval start plus the final end.
    if not rec.empty:
        xticks = sorted(rec[COL_REL_START].unique())
        xticks.append(x_max)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=FONT_SIZE)

    ax.set_xlabel('Within-trial time (s) — interval start',
                  fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(_y_label(metric, value_type), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(f"{metric} — baseline vs recovery by condition  (band: {band_lbl})",
                 fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)
    # Many entries -- place the legend outside the axes so it never covers data.
    ax.legend(fontsize=FONT_SIZE, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    fig.tight_layout()

    if save:
        fname = f"baseline_comparison_bycond_{metric}_{value_type}_{error_band}{file_tag}"
        _save(fig, fname, figures_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
