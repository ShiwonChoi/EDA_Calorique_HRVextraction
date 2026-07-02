# Temp_binning functions
#
# Chunks a trial's cleaned RRI signal into sequential fixed-width bins and
# computes temporal HRV metrics per bin via get_temp_metrics, baseline-
# referenced against the baseline TRIAL's whole-trial metric value
# (baseline_temp_raw) using build_result_row's existing diff/pct_change/
# log_ratio derivation.

import numpy as np
import pandas as pd

from lib.config import OUTPUT_COLUMNS
from lib.Metric_extraction.HRV_temp_extract import get_temp_metrics, phase_windows, label_bin
from lib.Metric_extraction.HRV_df import build_result_row


def _empty_output():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def bin_temp_30s(intervals_clean, beat_times_clean, trial, condition,
                 task_interval, df_events_t, participant_id,
                 baseline_temp_raw, sample_size, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and compute temporal HRV metrics per bin, baseline-referenced
    against baseline_temp_raw[metric_name] (the baseline TRIAL's whole-trial
    value) via build_result_row.

    Each bin's task_moment is the recording phase (anticipation/task/recovery)
    whose event window contains the bin center, or 'unclassified' if it falls
    in a gap. recording_type is always 'interval'.

    time_interval_rel_start / abs_start: bin start (relative to task_start / absolute).
    time_interval_rel_end   / abs_end:   bin end   (relative to task_start / absolute).
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
        bin_start   = t
        bin_end     = min(t + bin_width, task_end)
        bin_center  = (bin_start + bin_end) / 2
        rel_start   = bin_start - task_start
        rel_end     = bin_end   - task_start
        task_moment = label_bin(bin_center, phases)

        metrics = get_temp_metrics(intervals_clean, beat_times_clean,
                                   t_start=bin_start, t_end=bin_end)

        for metric_name, metric_value in metrics.items():
            baseline_mean = baseline_temp_raw.get(metric_name, float('nan'))
            rows.extend(build_result_row(
                participant_id=participant_id,
                trial=trial,
                condition=condition,
                time_interval_rel_start=rel_start,
                time_interval_abs_start=bin_start,
                time_interval_rel_end=rel_end,
                time_interval_abs_end=bin_end,
                task_moment=task_moment,
                recording_type='interval',
                metric_name=metric_name,
                metric_value=metric_value,
                baseline_mean=baseline_mean,
                sample_size=sample_size,
            ))

        t += bin_width

    if not rows:
        return _empty_output()

    return pd.DataFrame(rows)[OUTPUT_COLUMNS]
