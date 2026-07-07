# VAS (subjective stress) extraction
#
# The touch_data_*.csv file is a continuous recording of a subjective-stress
# VAS rating (`position`, 0-1, scaled ×100 to a 0-100 score). Its own clock
# (`elapsed_s`) is zeroed to the START OF VAS RECORDING, not to Shimmer
# connection, so it must be shifted onto the shimmer-connected timeline used
# everywhere else in the pipeline:
#
#     time_since_connected_s = elapsed_s + touchslider_recording_start_ms/1000
#
# where touchslider_recording_start is an event in the aligned event log.
# Once shifted, VAS mean/median/std are computed over the SAME per-trial task
# windows and 30-s bins as the temporal HRV metrics, and baseline-referenced
# (raw / diff / pct_change / log_ratio) via build_result_row.

import numpy as np
import pandas as pd
from pathlib import Path

from lib.config import OUTPUT_COLUMNS
from lib.Metric_extraction.HRV_temp_extract import phase_windows, label_bin
from lib.Metric_extraction.HRV_df import build_result_row


def get_vas_recording_offset(participant_path):
    """
    Return the shimmer-connected time (seconds) at which VAS recording began,
    read from the `touchslider_recording_start` row of the aligned event log.

    clean_events drops this event (it is not in ALLCOND), so it is read here
    straight from event_log_*_aligned.csv rather than from the cleaned
    df_events. Returns None if the log or the marker is missing.
    """
    participant_path = Path(participant_path)
    event_files = sorted(participant_path.glob("event_log_*_aligned.csv"))
    if len(event_files) != 1:
        print(f"  VAS: expected one event_log_*_aligned.csv in {participant_path}, "
              f"found {len(event_files)} — skipping VAS")
        return None

    ev = pd.read_csv(event_files[0])
    hit = ev.loc[ev["event_label"] == "touchslider_recording_start", "shimmer_device_ms"]
    if hit.empty:
        print(f"  VAS: no touchslider_recording_start in {event_files[0].name} — skipping VAS")
        return None

    return float(hit.iloc[0]) / 1000.0


def load_touch_vas(participant_path, recording_start_s):
    """
    Load the one touch_data_*.csv for this participant and return a DataFrame
    with:
        time_seconds — shimmer-connected time (elapsed_s + recording_start_s)
        vas          — VAS score, position × 100 (0-100 scale)

    Returns None if no (or more than one) touch_data file is found, or if
    recording_start_s is None.
    """
    if recording_start_s is None:
        return None

    participant_path = Path(participant_path)
    touch_files = sorted(participant_path.glob("touch_data_*.csv"))
    if len(touch_files) != 1:
        print(f"  VAS: expected one touch_data_*.csv in {participant_path}, "
              f"found {len(touch_files)} — skipping VAS")
        return None

    df = pd.read_csv(touch_files[0])
    if "elapsed_s" not in df.columns or "position" not in df.columns:
        print(f"  VAS: {touch_files[0].name} missing elapsed_s/position — skipping VAS")
        return None

    out = pd.DataFrame({
        "time_seconds": df["elapsed_s"].astype(float) + recording_start_s,
        "vas":          df["position"].astype(float) * 100.0,
    }).sort_values("time_seconds").reset_index(drop=True)

    print(f"  VAS: {len(out)} samples loaded "
          f"[{out['time_seconds'].min():.1f}s – {out['time_seconds'].max():.1f}s]")
    return out


def get_vas_metrics(df_touch, t_start, t_end):
    """
    VAS mean / median / std over [t_start, t_end] (seconds, shimmer-connected).

    Returns
    -------
    metrics : dict — VAS_mean, VAS_median, VAS_std (std is NaN with <2 samples).
              Empty dict if the window contains no samples.
    n       : int  — number of VAS samples in the window.
    """
    m = (df_touch["time_seconds"] >= t_start) & (df_touch["time_seconds"] <= t_end)
    v = df_touch.loc[m, "vas"].to_numpy(dtype=float)
    if v.size < 1:
        return {}, 0

    metrics = {
        "VAS_mean":   float(np.mean(v)),
        "VAS_median": float(np.median(v)),
        "VAS_std":    float(np.std(v, ddof=1)) if v.size >= 2 else float("nan"),
    }
    return metrics, int(v.size)


def _empty_output():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def bin_vas_30s(df_touch, trial, condition, task_interval, df_events_t,
                participant_id, baseline_vas_raw, bin_width=30):
    """
    Chunk the task window into sequential fixed-width bins (last bin may be
    shorter) and compute VAS mean/median/std per bin, baseline-referenced
    against baseline_vas_raw[metric_name] via build_result_row.

    Bin edges, phase labelling (task_moment) and recording_type='interval'
    match bin_temp_30s so VAS bins align with the temporal HRV bins.
    """
    task_start, task_end = task_interval

    if pd.isna(task_start) or pd.isna(task_end):
        print(f"  No {trial} task window — emitting no {trial} 30s-bin VAS rows")
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

        metrics, n = get_vas_metrics(df_touch, bin_start, bin_end)
        sample_size = f"{n}"

        for metric_name, metric_value in metrics.items():
            baseline_mean = baseline_vas_raw.get(metric_name, float("nan"))
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
