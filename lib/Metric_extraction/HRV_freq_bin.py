# Freq_binning functions
#
# Aggregates a trial's long-format band-power DataFrame (band_df, from
# HRV_freq_extract.band_results_to_df) into:
#   - one whole-trial mean per band/value_type (bin_totalbandpower)
#   - sequential fixed-width bins per band/value_type (bin_bandpower_30s)
#
# band_df carries raw and (for stim trials) diff/pct_change/log_ratio
# value_types, baseline-corrected upstream in run_cwt_task/per_frequency_correction
# against the baseline TRIAL's per-frequency power. These functions only average —
# no re-derivation. Frequency and temporal HRV use the same Value_type set
# (VALUE_TYPES in config.py) though the frequency corrections are done
# per-frequency before band integration, unlike temporal HRV which derives them
# directly from beat-to-beat metrics.
#
# Baseline trial rows emit diff/pct_change/log_ratio = 0.0 (same convention as
# temporal HRV) so that all Value_types are present uniformly across conditions.

import numpy as np
import pandas as pd

from lib.config import OUTPUT_COLUMNS, VALUE_TYPES
from lib.Metric_extraction.HRV_temp_extract import phase_windows, label_bin


def _mean_band_power_by_valuetype(df, real_start, real_end, condition):
    """
    Mean band power within [real_start, real_end) per (band, value_type).

    Always iterates over all VALUE_TYPES. For the baseline condition, missing
    value_types (diff / pct_change / log_ratio — not produced by the upstream
    per-frequency correction because baseline is the reference itself) are
    filled with 0.0; for stim trials a missing value_type yields NaN.

    Returns
    -------
    dict  {(band, value_type): mean_value, ...}
    """
    mask = (df['time_seconds'] >= real_start) & (df['time_seconds'] < real_end)
    sub = df[mask]
    is_baseline = (condition == 'baseline')

    out = {}
    for band in df['band'].unique():
        for vt in VALUE_TYPES:
            vals = sub.loc[(sub['band'] == band) & (sub['value_type'] == vt), 'power']
            if len(vals) == 0:
                out[(band, vt)] = 0.0 if (is_baseline and vt != 'raw') else np.nan
            else:
                out[(band, vt)] = float(vals.mean())
    return out


def _make_row(participant_id, trial, condition,
              time_interval_rel_start, time_interval_abs_start,
              time_interval_rel_end,   time_interval_abs_end,
              task_moment, recording_type,
              band, value_type, power,
              sample_size, status="SUCCESS", error=None):
    return {
        'participant':              participant_id,
        'trial':                    trial,
        'condition':                condition,
        'time_interval_rel_start':  time_interval_rel_start,
        'time_interval_abs_start':  time_interval_abs_start,
        'time_interval_rel_end':    time_interval_rel_end,
        'time_interval_abs_end':    time_interval_abs_end,
        'task_moment':              task_moment,
        'recording_type':           recording_type,
        'Metric':                   band,
        'Value_type':               value_type,
        'Value':                    power,
        'sample_size':              sample_size,
        'status':                   status,
        'error':                    error,
    }


def _empty_output():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def bin_totalbandpower(bandpower, trial, condition, task_interval,
                       participant_id, sample_size):
    """
    Mean band power over the entire task window — one row per (band, value_type).
    recording_type = 'total'.
    task_moment    = 'baseline' for the baseline trial, 'total' for stim trials.
    rel_start = 0, abs_start = task_start; rel_end = task duration, abs_end = task_end.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial}_total rows")
        return _empty_output()

    task_moment = 'baseline' if condition == 'baseline' else 'total'
    task_duration = task_end - task_start

    print(f"Extracting {trial}_total metrics from {task_start:.2f} to {task_end:.2f}s")
    total_metrics = _mean_band_power_by_valuetype(bandpower, task_start, task_end, condition)

    rows = [
        _make_row(
            participant_id=participant_id,
            trial=trial,
            condition=condition,
            time_interval_rel_start=0.0,
            time_interval_abs_start=task_start,
            time_interval_rel_end=task_duration,
            time_interval_abs_end=task_end,
            task_moment=task_moment,
            recording_type='total',
            band=band,
            value_type=vt,
            power=mean_val,
            sample_size=sample_size,
        )
        for (band, vt), mean_val in total_metrics.items()
    ]

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def bin_bandpower_30s(bandpower, trial, condition, task_interval, df_events_t,
                      participant_id, sample_size, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and average each (band, value_type) within each bin.
    recording_type = 'interval'.
    task_moment is the recording phase whose window contains the bin center
    (anticipation / task / recovery / unclassified).

    rel_start / abs_start: bin start relative to task_start / absolute.
    rel_end   / abs_end:   bin end   relative to task_start / absolute.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial} 30s-bin rows")
        return _empty_output()

    phases = phase_windows(df_events_t)

    rows = []
    t = task_start
    while t < task_end:
        bin_start   = t
        bin_end     = min(t + bin_width, task_end)
        bin_center  = (bin_start + bin_end) / 2
        rel_start   = bin_start - task_start
        rel_end     = bin_end   - task_start
        task_moment = label_bin(bin_center, phases)

        bin_metrics = _mean_band_power_by_valuetype(bandpower, bin_start, bin_end, condition)

        for (band, vt), mean_val in bin_metrics.items():
            rows.append(_make_row(
                participant_id=participant_id,
                trial=trial,
                condition=condition,
                time_interval_rel_start=rel_start,
                time_interval_abs_start=bin_start,
                time_interval_rel_end=rel_end,
                time_interval_abs_end=bin_end,
                task_moment=task_moment,
                recording_type='interval',
                band=band,
                value_type=vt,
                power=mean_val,
                sample_size=sample_size,
            ))

        t += bin_width

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[OUTPUT_COLUMNS]
