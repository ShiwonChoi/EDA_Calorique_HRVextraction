"""
Boxplot of a single "total"-window metric across the cohort.

The processed result CSVs (processed_ppg_results_{temp,freq,gsr}.csv) share the
unified schema defined in lib.config.OUTPUT_COLUMNS. Every row is one metric
value for one participant, in one trial, under one condition, over one time
window. ``recording_type`` distinguishes the whole-task summary ('total') from
the 30 s sub-windows ('interval').

This module plots, for a chosen ``Metric`` and ``Value_type``, the distribution
of the whole-task ('total') value across participants — one group of boxes per
trial, one box per condition/group inside each trial.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Stable colour per experimental condition/group so the same colour always
# means the same thing across the temp/freq/gsr figures.
CONDITION_COLORS = {
    'baseline': '#9e9e9e',
    'LC':       '#4c78a8',   # Left  / Cold
    'LW':       '#e45756',   # Left  / Warm
    'RC':       '#72b7b2',   # Right / Cold
    'RW':       '#f58518',   # Right / Warm
}
_FALLBACK_COLOR = '#bab0ac'


def plot_total_metric_boxplot(
    csv_path,
    metric,
    value_type="raw",
    group_col="condition",
    trial_col="trial",
    ax=None,
    box_width=0.7,
    save_path=None,
    show=True,
):
    """
    Boxplot the whole-task ('total') value of one metric, grouped by trial and
    split by condition/group.

    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to one of the processed_ppg_results_{temp,freq,gsr}.csv files.
    metric : str
        Value of the ``Metric`` column to plot (e.g. 'mean_HR', 'LF',
        'Tonic_SCL_mean'). Must exist in the file.
    value_type : str, default 'raw'
        Value of the ``Value_type`` column to plot: one of
        'raw', 'diff', 'pct_change', 'log_ratio'.
    group_col : str, default 'condition'
        Column whose levels become the coloured boxes within each trial.
    trial_col : str, default 'trial'
        Column whose levels become the x-axis groups.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created when omitted.
    box_width : float, default 0.7
        Fraction of each trial slot occupied by that trial's boxes.
    save_path : str | pathlib.Path, optional
        If given, the figure is written here (dpi=150, tight bbox).
    show : bool, default True
        Call plt.show() at the end (ignored when an external ``ax`` is passed).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis the boxplot was drawn on.
    """
    df = pd.read_csv(csv_path)

    # Keep only whole-task rows for the requested metric / value type, and drop
    # failures or non-numeric values so the boxes reflect real measurements.
    mask = (
        (df["recording_type"] == "total")
        & (df["Metric"] == metric)
        & (df["Value_type"] == value_type)
    )
    sub = df.loc[mask].copy()
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub = sub.dropna(subset=["Value"])

    if sub.empty:
        raise ValueError(
            f"No 'total' rows for Metric={metric!r}, Value_type={value_type!r} "
            f"in {csv_path}. Available metrics: "
            f"{sorted(df['Metric'].unique())}"
        )

    trials = sorted(sub[trial_col].unique())
    groups = sorted(sub[group_col].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(trials)), 5))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    n_groups = len(groups)
    slot = box_width / n_groups                      # width per group box
    offsets = [(-box_width / 2) + slot * (i + 0.5) for i in range(n_groups)]

    legend_handles = {}
    for gi, group in enumerate(groups):
        color = CONDITION_COLORS.get(group, _FALLBACK_COLOR)
        data, positions = [], []
        for ti, trial in enumerate(trials):
            vals = sub.loc[
                (sub[trial_col] == trial) & (sub[group_col] == group), "Value"
            ].values
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(ti + offsets[gi])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=slot * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.55),
        )

        # Overlay the individual participant points for transparency.
        for vals, pos in zip(data, positions):
            jitter = (pd.Series(range(len(vals))) - (len(vals) - 1) / 2)
            jitter = jitter * (slot * 0.12) / max(len(vals), 1)
            ax.scatter(
                pos + jitter.values, vals,
                s=14, color=color, edgecolor="white", linewidth=0.4,
                zorder=3,
            )
        legend_handles[group] = bp["boxes"][0]

    ax.set_xticks(range(len(trials)))
    ax.set_xticklabels(trials)
    ax.set_xlabel(trial_col)
    ax.set_ylabel(f"{metric} ({value_type})")
    ax.set_title(f"{metric} — whole-task value by {group_col} per {trial_col}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(
        legend_handles.values(), legend_handles.keys(),
        title=group_col, frameon=False,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
    )
    if created_fig:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and created_fig:
        plt.show()

    return ax


def plot_total_metric_boxplot_by_trial(
    csv_path,
    metric,
    value_type="raw",
    trial_col="trial",
    group_col="condition",
    ax=None,
    box_color="#4c78a8",
    save_path=None,
    show=True,
):
    """
    Boxplot the whole-task ('total') value of one metric, one box per trial,
    pooling all conditions (LC/LW/RC/RW/baseline) together.

    Same filtering as :func:`plot_total_metric_boxplot` — 'total' rows for the
    requested ``metric`` and ``value_type`` — but the condition/group split is
    ignored for the box itself, so each trial is a single distribution over
    every participant and condition in that trial. The overlaid individual
    points, however, are coloured by ``group_col`` (via ``CONDITION_COLORS``)
    so participants' conditions remain visible.

    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to one of the processed_ppg_results_{temp,freq,gsr}.csv files.
    metric : str
        Value of the ``Metric`` column to plot (e.g. 'mean_HR', 'LF').
    value_type : str, default 'raw'
        Value of the ``Value_type`` column: 'raw', 'diff', 'pct_change',
        'log_ratio'.
    trial_col : str, default 'trial'
        Column whose levels become the x-axis boxes.
    group_col : str, default 'condition'
        Column used to colour the individual point overlay via
        ``CONDITION_COLORS``. Does not affect box grouping.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created when omitted.
    box_color : str, default '#4c78a8'
        Fill colour for every box.
    save_path : str | pathlib.Path, optional
        If given, the figure is written here (dpi=150, tight bbox).
    show : bool, default True
        Call plt.show() at the end (ignored when an external ``ax`` is passed).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis the boxplot was drawn on.
    """
    df = pd.read_csv(csv_path)

    mask = (
        (df["recording_type"] == "total")
        & (df["Metric"] == metric)
        & (df["Value_type"] == value_type)
    )
    sub = df.loc[mask].copy()
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub = sub.dropna(subset=["Value"])

    if sub.empty:
        raise ValueError(
            f"No 'total' rows for Metric={metric!r}, Value_type={value_type!r} "
            f"in {csv_path}. Available metrics: "
            f"{sorted(df['Metric'].unique())}"
        )

    trials = sorted(sub[trial_col].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(trials)), 5))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    data, positions, kept_trials, groups_per_trial = [], [], [], []
    for ti, trial in enumerate(trials):
        trial_rows = sub.loc[sub[trial_col] == trial]
        vals = trial_rows["Value"].values
        if len(vals) == 0:
            continue
        data.append(vals)
        positions.append(ti)
        kept_trials.append(trial)
        groups_per_trial.append(trial_rows[group_col].values)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color=box_color),
        capprops=dict(color=box_color),
        boxprops=dict(facecolor=box_color, edgecolor=box_color, alpha=0.55),
    )

    # Overlay individual participant points, coloured by condition/group for
    # transparency into the pooled box.
    legend_handles = {}
    for vals, pos, point_groups in zip(data, positions, groups_per_trial):
        jitter = (pd.Series(range(len(vals))) - (len(vals) - 1) / 2)
        jitter = jitter * 0.03
        point_colors = [
            CONDITION_COLORS.get(g, _FALLBACK_COLOR) for g in point_groups
        ]
        ax.scatter(
            pos + jitter.values, vals,
            s=16, color=point_colors, edgecolor="white", linewidth=0.4,
            zorder=3,
        )
        for g in pd.unique(point_groups):
            legend_handles.setdefault(
                g, plt.Line2D(
                    [], [], marker="o", linestyle="",
                    markerfacecolor=CONDITION_COLORS.get(g, _FALLBACK_COLOR),
                    markeredgecolor="white", markersize=6,
                )
            )

    ax.set_xticks(range(len(kept_trials)))
    ax.set_xticklabels(kept_trials)
    ax.set_xlabel(trial_col)
    ax.set_ylabel(f"{metric} ({value_type})")
    ax.set_title(f"{metric} — whole-task value per {trial_col} (all conditions)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    if legend_handles:
        ax.legend(
            legend_handles.values(), legend_handles.keys(),
            title=group_col, frameon=False,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
        )
    if created_fig:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and created_fig:
        plt.show()

    return ax
