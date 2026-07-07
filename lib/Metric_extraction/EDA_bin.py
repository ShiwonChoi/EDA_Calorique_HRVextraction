# EDA (electrodermal activity) binning
#
# Chunks a trial's task window into sequential fixed-width bins and computes
# tonic/phasic EDA metrics per bin via get_eda_metrics, baseline-referenced
# against the baseline trial's whole-trial value (baseline_eda_raw) using
# build_result_row's existing diff/pct_change/log_ratio derivation.
#
# Mirrors VAS_extract.py's bin_vas_30s: results_gsr (from
# preprocess_visualize_gsr) is computed ONCE for the whole continuous
# recording, and this function only windows into it per bin -- it does not
# reload or reprocess the signal.

import pandas as pd

from lib.config import OUTPUT_COLUMNS
from lib.Metric_extraction.HRV_temp_extract import phase_windows, label_bin
from lib.Metric_extraction.HRV_df import build_result_row
from lib.Metric_extraction.EDA_temp_extract import get_eda_metrics
from lib.GSR_extract.gsr_preprocess import masked_sample_counts


def _empty_output():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def bin_eda_30s(results_gsr, trial, condition, task_interval, df_events_t,
                 participant_id, baseline_eda_raw, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and compute tonic/phasic EDA metrics per bin, baseline-referenced
    against baseline_eda_raw[metric_name] (the baseline trial's whole-trial
    value) via build_result_row.

    Each bin's task_moment is the recording phase whose event window contains
    the bin center (anticipation/task/recovery/unclassified), matching
    bin_temp_30s/bin_vas_30s. recording_type is always 'interval'.

    sample_size is computed per bin from results_gsr via masked_sample_counts
    ("<n_clean GSR samples> / <n_raw GSR samples>" in that bin), unlike
    bin_temp_30s/bin_vas_30s which reuse one whole-trial value across all bins
    -- results_gsr's artifact mask makes a real per-bin count cheap and more
    accurate here.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window -- emitting no {trial} 30s-bin EDA rows")
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

        metrics = get_eda_metrics(results_gsr, t_start=bin_start, t_end=bin_end)
        n_clean, n_raw = masked_sample_counts(results_gsr, bin_start, bin_end)
        sample_size = f"{n_clean} / {n_raw}"

        for metric_name, metric_value in metrics.items():
            baseline_mean = baseline_eda_raw.get(metric_name, float('nan'))
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
