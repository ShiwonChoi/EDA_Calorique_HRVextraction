import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lib.PPG_extract.ppg_preprocess import clean_events, assign_trial_condition

def load_ppg(participant_path, show=True):
    """
    Load the shimmer PPG and event-log CSVs for a single participant's session.

    Expects a flat layout directly under participant_path: exactly one
    shimmer_*.csv and one event_log_*_aligned.csv, covering the whole
    continuous recording (baseline + all blocks + post-recovery in one file).

    Args:
        participant_path: Path to the SC_## participant folder.
        show: If True, print a loading summary.

    Returns:
        df_ppg         : Shimmer DataFrame for the whole session.
        df_events      : Event-log DataFrame with 'trial'/'condition' columns
                          derived by assign_trial_condition.
        participant_id : Folder name (e.g. "SC_18").
    """
    participant_path = Path(participant_path)
    participant_id = participant_path.name

    shimmer_files = sorted(participant_path.glob("shimmer_*.csv"))
    if len(shimmer_files) != 1:
        raise ValueError(
            f"Expected exactly one shimmer_*.csv in {participant_path}, found {len(shimmer_files)}"
        )
    event_files = sorted(participant_path.glob("event_log_*_aligned.csv"))
    if len(event_files) != 1:
        raise ValueError(
            f"Expected exactly one event_log_*_aligned.csv in {participant_path}, found {len(event_files)}"
        )

    # Shimmer CSV: row 0 = column names, rows 1-2 = unit labels (skip)
    df_ppg = pd.read_csv(shimmer_files[0], header=0, skiprows=[1, 2])
    df_ppg["participant"] = participant_id

    df_events = pd.read_csv(event_files[0])
    df_events = df_events.rename(columns={
        "event_label":      "event_type",
        "shimmer_device_ms": "time_since_connected_ms",
        "trial":            "raw_block_index",
        "block":            "raw_block_label",
    })
    df_events["participant"] = participant_id
    df_events = assign_trial_condition(df_events)

    if show:
        print(
            f"  Loaded {participant_id} | shimmer rows: {len(df_ppg)} "
            f"| event rows: {len(df_events)}"
        )

    return df_ppg, df_events, participant_id


def find_bad_segments(df):
    x = df['time_seconds'].values
    timestamp_differences = np.diff(x)
    dropped_s = np.sum(timestamp_differences > 0.02)
    print("The number of dropped samples is " + str(dropped_s))

    bad_timestamps = timestamp_differences > 0.02
    shiftright = np.append(np.array(False), bad_timestamps)
    shiftleft = np.append(bad_timestamps, np.array(False))
    badsegments = shiftright + shiftleft
    large_ts = timestamp_differences[bad_timestamps]

    return badsegments


def resample_signal(df, badsegments, fs_new):

    time_idx  = df.index
    period_str = str(1000/fs_new)+'ms'

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Resample numeric signal columns; akima preserves PPG waveform shape without overshoot
    df_resamp     = df.copy().iloc[:, 1:]
    df_resamp_nan = df_resamp.resample(period_str).mean(numeric_only=True)
    df_resamp     = df_resamp_nan.interpolate('akima')

    badsegments_df   = pd.DataFrame(badsegments, time_idx)
    badsegments_df_r = badsegments_df.resample(period_str).max()
    badsegments_r    = badsegments_df_r.fillna(0).values.squeeze().astype(bool)

    # Reindex categorical columns (trial, condition, participant) onto the new grid
    if cat_cols:
        df_resamp[cat_cols] = df[cat_cols].reindex(df_resamp.index, method='ffill')

    Time_s_resamp = df_resamp.index.total_seconds()
    df_resamp['time_seconds'] = Time_s_resamp

    return df_resamp, fs_new, badsegments_r


def check_signals_ppg(df, badsegments, event_dict):
    markers = event_dict.keys()

    task_markers = [marker for marker in markers if marker.endswith('start')]
    task_onsets = [event_dict[marker] for marker in task_markers]

    conditions = [marker.split('_')[1] for marker in task_markers]

    task_end_markers = [marker for marker in markers if marker.endswith('end')]
    task_offsets = [event_dict[marker] for marker in task_end_markers]

    signal_PPG = df['PPG'].values
    signal_GSR = df['GSR'].values

    x = df.index.astype('timedelta64[ms]')

    # look at all the resampled signals
    fig, axs = plt.subplots(2, 1, sharex=True)

    # PPG
    axs[0].plot(x, signal_PPG)
    bad_signal = signal_PPG.copy().squeeze()
    bad_signal[~badsegments] = np.nan
    axs[0].plot(x, bad_signal, 'r')
    axs[0].title.set_text('PPG')
    axs[0].set_ylabel('millivolts (uV)')
    ymin = axs[0].axes.get_ylim()[0]
    ymax = axs[0].axes.get_ylim()[1]
    axs[0].vlines(task_onsets, ymin=ymin*1.2, ymax=ymax*1.2, colors='k')
    axs[0].vlines(task_offsets, ymin=ymin*1.2, ymax=ymax*1.2, colors='r')

    for c_idx, cond in enumerate(conditions):
        axs[0].annotate(cond, xy=(task_onsets[c_idx], ymax), xytext=(task_onsets[c_idx] + 30, ymax * 1.1))

    # GSR
    axs[1].plot(x, signal_GSR)
    bad_signal = signal_GSR.copy().squeeze()
    bad_signal[~badsegments] = np.nan
    axs[1].plot(x, bad_signal, 'r')
    axs[1].title.set_text('GSR')
    axs[1].set_ylabel('millivolts (uV)')
    ymin = axs[1].axes.get_ylim()[0]
    ymax = axs[1].axes.get_ylim()[1]
    axs[1].vlines(task_onsets, ymin=ymin*1.2, ymax=ymax*1.2, colors='k')
    axs[1].vlines(task_offsets, ymin=ymin*1.2, ymax=ymax*1.2, colors='r')

    for c_idx, cond in enumerate(conditions):
        axs[1].annotate(cond, xy=(task_onsets[c_idx], ymax), xytext=(task_onsets[c_idx] + 30, ymax * 1.1))

    plt.xlabel('time_seconds')


def tag_ppg_trial_condition(df_ppg, df_events, participant_id):
    """
    Stamp 'participant'/'trial'/'condition' onto each df_ppg row by matching
    its 'rel_zero_ref (ms)' timestamp against the [min, max] time_since_connected_ms
    window of each trial in df_events (same clock origin — shimmer-connect zero).

    Rows outside every trial's window (e.g. shimmer kept recording after
    experiment_end) are left untagged (NaN) and are naturally excluded once
    downstream code slices df_ppg by trial.
    """
    df_ppg = df_ppg.copy()
    df_ppg["participant"] = participant_id
    df_ppg["trial"]       = pd.NA
    df_ppg["condition"]   = pd.NA

    bounds = df_events.dropna(subset=["trial"]).groupby("trial").agg(
        lo=("time_since_connected_ms", "min"),
        hi=("time_since_connected_ms", "max"),
        condition=("condition", "first"),
    )

    rel_ms = df_ppg["rel_zero_ref (ms)"]
    for trial, row in bounds.iterrows():
        mask = (rel_ms >= row["lo"]) & (rel_ms <= row["hi"])
        df_ppg.loc[mask, "trial"]     = trial
        df_ppg.loc[mask, "condition"] = row["condition"]

    return df_ppg


def load_and_clean_ppg(participants_path, show=False):
    """
    Args:
        participants_path : Path to the SC_## participant folder.
        show               : If True, display raw signal quality plot after loading.
    """
    df_ppg_raw, df_events, participant_id = load_ppg(participants_path, show=show)
    df_events = clean_events(df_events)

    df_ppg = df_ppg_raw.copy()
    df_ppg = df_ppg.rename(columns={'Internal ADC A13': 'PPG'})

    ts  = df_ppg["Time Stamp"].astype(float)
    rel = ts - ts.iloc[0]
    df_ppg["rel_zero_ref (ms)"]  = rel
    df_ppg["abs_zero_ref (ms)"]  = rel
    df_ppg.index            = pd.to_timedelta(rel, unit='ms')
    df_ppg["time_seconds"]  = df_ppg["rel_zero_ref (ms)"] / 1000

    df_ppg = tag_ppg_trial_condition(df_ppg, df_events, participant_id)

    # Remove duplicate recording
    dup_mask = df_ppg.index.duplicated(keep='first')
    n_dup    = int(dup_mask.sum())
    if n_dup:
        print(f"  {participant_id}: {n_dup} duplicate timestamps removed")
        df_ppg = df_ppg[~dup_mask]

    # Bad segments & resampling
    badsegments             = find_bad_segments(df_ppg)
    df_ppg, fs, badsegments = resample_signal(df_ppg, badsegments, fs_new=250)

    # Replace interpolated values with exact regular-grid times
    rel_ms                 = df_ppg.index.total_seconds() * 1000
    df_ppg["rel_zero_ref (ms)"] = rel_ms
    df_ppg["abs_zero_ref (ms)"] = rel_ms
    df_ppg["time_seconds"] = rel_ms / 1000

    if show:
        check_signals_ppg(df_ppg, badsegments, df_events)
        plt.show()

    return df_ppg, df_events, fs, badsegments, participant_id
