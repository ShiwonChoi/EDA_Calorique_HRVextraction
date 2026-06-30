# Freq_binning functions
#
# Aggregates a trial's long-format band-power DataFrame (band_df, from
# HRV_freq_extract.band_results_to_df) into:
#   - one whole-trial mean per band/value_type (bin_totalbandpower)
#   - sequential fixed-width bins per band/value_type (bin_bandpower_30s)
#
# band_df already carries raw and (for stim trials) diff/pct_change/log_ratio
# value_types, baseline-corrected upstream in run_cwt_task/per_frequency_correction
# against the baseline TRIAL's own per-frequency power. These functions only
# average — no re-derivation.


# Libraries
import numpy as np
import pandas as pd

from lib.Metric_extraction.HRV_temp_extract import phase_windows, label_bin


def _mean_band_power_by_valuetype(df, real_start, real_end):
    """
    Mean band power within [real_start, real_end), per (band, value_type).

    Returns
    -------
    dict
        {(band, value_type): {'mean': scalar, 'n_samples': int}, ...}
    """
    mask = (df['time_seconds'] >= real_start) & (df['time_seconds'] < real_end)
    sub = df[mask]

    out = {}
    for band in df['band'].unique():
        for vt in df['value_type'].unique():
            vals = sub.loc[(sub['band'] == band) & (sub['value_type'] == vt), 'power']
            if len(vals) == 0:
                out[(band, vt)] = {'mean': np.nan, 'n_samples': 0}
            else:
                out[(band, vt)] = {'mean': float(vals.mean()), 'n_samples': len(vals)}
    return out


def _make_row(participant_id, trial, condition, task_moment,
              time_interval_relative, time_center_plot,
              band, value_type, power, n_samples, status="SUCCESS"):
    return {
        'participant_id':         participant_id,
        'trial':                  trial,
        'condition':              condition,
        'task_moment':            task_moment,
        'time_interval_relative': time_interval_relative,
        'time_center_plot':       time_center_plot,
        'Metric':                 band,
        'Value_type':             value_type,
        'Value':                  power,
        'n_samples':              n_samples,
        'status':                 status,
    }


_OUTPUT_COLUMNS = [
    'participant_id', 'trial', 'condition', 'task_moment',
    'time_interval_relative', 'time_center_plot',
    'Metric', 'Value_type', 'Value', 'n_samples', 'status',
]


def _empty_output():
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


# Functions for binning Time-Freq DataFrame
# ----------------------------------------------------------------------------------
def bin_totalbandpower(bandpower, trial, condition, task_interval, participant_id):
    """
    Mean band power over the entire task window — one row per (band, value_type)
    present in bandpower. Averages each value_type directly — no re-derivation.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial}_total rows")
        return _empty_output()

    print(f"Extracting {trial}_total metrics from {task_start:.2f} to {task_end:.2f}s")
    total_metrics = _mean_band_power_by_valuetype(bandpower, task_start, task_end)

    rows = [
        _make_row(
            participant_id=participant_id, trial=trial, condition=condition,
            task_moment=f"{trial}_total",
            time_interval_relative=0, time_center_plot=0,
            band=band, value_type=vt, power=vals['mean'], n_samples=vals['n_samples'],
        )
        for (band, vt), vals in total_metrics.items()
    ]

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[_OUTPUT_COLUMNS]


def bin_bandpower_30s(bandpower, trial, condition, task_interval, df_events_t,
                      participant_id, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and average each (band, value_type) within each bin. Averages
    directly — no re-derivation.

    Each bin is labeled with the recording phase (anticipation/task/recovery)
    whose event window its center falls within, derived from this trial's
    actual event markers — 'unclassified' if it falls in a gap between phases.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial} 30s-bin rows")
        return _empty_output()

    phases = phase_windows(df_events_t)

    rows = []
    t = task_start
    while t < task_end:
        bin_start = t
        bin_end   = min(t + bin_width, task_end)
        bin_center = (bin_start + bin_end) / 2
        rel_start = bin_start - task_start

        task_moment = label_bin(bin_center, phases)

        bin_metrics = _mean_band_power_by_valuetype(bandpower, bin_start, bin_end)

        for (band, vt), vals in bin_metrics.items():
            rows.append(_make_row(
                participant_id=participant_id, trial=trial, condition=condition,
                task_moment=task_moment,
                time_interval_relative=rel_start,
                time_center_plot=rel_start + (bin_end - bin_start) / 2,
                band=band, value_type=vt, power=vals['mean'], n_samples=vals['n_samples'],
            ))

        t += bin_width

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[_OUTPUT_COLUMNS]
