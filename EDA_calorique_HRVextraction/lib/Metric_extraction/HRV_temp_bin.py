# Temp_binning functions
#
# Chunks a trial's cleaned RRI signal into sequential fixed-width bins and
# computes temporal HRV metrics per bin via get_temp_metrics, baseline-
# referenced against the baseline TRIAL's whole-trial metric value
# (baseline_temp_raw) using build_result_row's existing diff/pct_change/
# log_ratio derivation — no separate derivation logic duplicated here.


# Libraries
import numpy as np
import pandas as pd

from lib.Metric_extraction.HRV_temp_extract import get_temp_metrics, phase_windows, label_bin
from lib.Metric_extraction.HRV_df import build_result_row


_OUTPUT_COLUMNS = [
    'participant_id', 'trial', 'condition', 'task_moment',
    'time_interval_relative', 'time_center_plot',
    'Metric', 'Value_type', 'Value', 'n_samples', 'status',
]


def _empty_output():
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


# Functions for binning the temporal HRV signal
# ----------------------------------------------------------------------------------
def bin_temp_30s(intervals_clean, beat_times_clean, trial, condition,
                 task_interval, df_events_t, participant_id,
                 baseline_temp_raw, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and compute temporal HRV metrics per bin, baseline-referenced
    against baseline_temp_raw[metric_name] (the baseline TRIAL's whole-trial
    value) via build_result_row.

    Each bin is labeled with the recording phase (anticipation/task/recovery)
    whose event window its center falls within, derived from this trial's
    actual event markers — 'unclassified' if it falls in a gap between phases.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial} 30s-bin temporal rows")
        return _empty_output()

    phases = phase_windows(df_events_t)
    beat_times_clean = np.asarray(beat_times_clean, dtype=float)

    rows = []
    t = task_start
    while t < task_end:
        bin_start = t
        bin_end   = min(t + bin_width, task_end)
        bin_center = (bin_start + bin_end) / 2
        rel_start = bin_start - task_start
        task_moment = label_bin(bin_center, phases)

        metrics = get_temp_metrics(intervals_clean, beat_times_clean,
                                   t_start=bin_start, t_end=bin_end)
        n_samples = int(np.sum((beat_times_clean >= bin_start) & (beat_times_clean <= bin_end)))

        for metric_name, metric_value in metrics.items():
            baseline_mean = baseline_temp_raw.get(metric_name, float('nan'))

            for result_row in build_result_row(
                participant_id, trial, condition,
                rel_time=float('nan'), abs_time=float('nan'),
                metric_name=metric_name, metric_value=metric_value,
                baseline_mean=baseline_mean,
            ):
                rows.append({
                    'participant_id':         participant_id,
                    'trial':                  trial,
                    'condition':              condition,
                    'task_moment':            task_moment,
                    'time_interval_relative': rel_start,
                    'time_center_plot':       rel_start + (bin_end - bin_start) / 2,
                    'Metric':                 result_row['Metric'],
                    'Value_type':             result_row['Value_type'],
                    'Value':                  result_row['Value'],
                    'n_samples':              n_samples,
                    'status':                 result_row['status'],
                })

        t += bin_width

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[_OUTPUT_COLUMNS]
