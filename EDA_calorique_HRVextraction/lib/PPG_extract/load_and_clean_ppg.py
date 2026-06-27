import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lib.PPG_extract.ppg_preprocess import match_events, clean_events

def load_ppg(participant_path, trial_filter=None, show=True):
    """
    Load shimmer PPG and events CSVs for a single participant.

    Scans the 'Stress measures' subfolder of participant_path for all
    shimmer_*.csv and events_*.csv files (Trial00/baseline, Trial01/condition,
    Trial02/condition), pairs them by shared filename suffix, and returns
    concatenated DataFrames across all three trials.

    Args:
        participant_path: Path to the SC_## participant folder.
        show: If True, print a loading summary per trial.

    Returns:
        df_ppg          : Combined shimmer DataFrame (all trials, with added
                          'participant', 'trial', 'condition' columns).
        df_events       : Combined events DataFrame (matching structure).
    """
    participant_path = Path(participant_path)
    stress_dir = participant_path / "Stress measures"

    if not stress_dir.exists():
        raise FileNotFoundError(
            f"No 'Stress measures' folder found in {participant_path}"
        )

    if trial_filter is not None and isinstance(trial_filter, str):
        trial_filter = {trial_filter}
    elif trial_filter is not None:
        trial_filter = set(trial_filter)

    shimmer_files = sorted(stress_dir.glob("shimmer_*.csv"))
    if not shimmer_files:
        raise ValueError(f"No shimmer CSV files found in {stress_dir}")

    # Build lookup: shared key → events file
    events_lookup = {
        ef.stem[len("events_"):]: ef
        for ef in stress_dir.glob("events_*.csv")
    }

    shimmer_dfs = []
    events_dfs = []

    for shimmer_file in shimmer_files:
        key = shimmer_file.stem[len("shimmer_"):]

        # Parse participant, trial, condition, timestamp from key parts
        # Format: P00#_Trial0#_<condition>_YYYYMMDD_HHMMSS
        parts = key.split("_")
        if len(parts) < 5:
            print(f"  Warning: unexpected shimmer filename format, skipping: {shimmer_file.name}")
            continue

        participant_id = parts[0]                     # e.g. "P003"
        trial         = parts[1]                     # e.g. "Trial01"
        condition     = "_".join(parts[2:-2])        # e.g. "baseline", "LW", "RW"

        if trial_filter is not None and trial not in trial_filter:
            continue

        # Shimmer CSV: row 0 = column names, rows 1-2 = unit labels (skip)
        df_shimmer = pd.read_csv(shimmer_file, header=0, skiprows=[1, 2])
        df_shimmer["participant"] = participant_id
        df_shimmer["trial"]      = trial
        df_shimmer["condition"]  = condition
        shimmer_dfs.append(df_shimmer)

        # Events CSV: row 0 is a comment line ("# First sample at: ...")
        if key in events_lookup:
            df_ev = pd.read_csv(events_lookup[key], comment="#")
            df_ev["participant"] = participant_id
            df_ev["trial"]      = trial
            df_ev["condition"]  = condition
            events_dfs.append(df_ev)
        else:
            print(f"  Warning: no matching events file for {shimmer_file.name}")

        if show:
            print(
                f"  Loaded {participant_id} | {trial} | {condition} "
                f"| shimmer rows: {len(df_shimmer)}"
            )

    df_ppg    = pd.concat(shimmer_dfs, ignore_index=True)
    df_events = pd.concat(events_dfs,  ignore_index=True) if events_dfs else pd.DataFrame()

    # Cross-check JSON vs CSV event files; raise on any mismatch
    event_check = match_events(participant_path)
    failed = [r for r in event_check if r["status"] != "OK"]
    if failed:
        lines = []
        for r in failed:
            tag = f"{r['participant']} | {r['trial']} | {r['condition']}"
            lines.append(f"  [{r['status']}] {tag}")
            for msg in r["mismatches"]:
                lines.append(f"    {msg}")
        raise ValueError(
            "Event file mismatch detected — aborting load:\n" + "\n".join(lines)
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


def load_and_clean_ppg(participants_path, trial_filter=None, show=False):
    """
    Args:
        participants_path : Path to the SC_## participant folder.
        trial_filter      : Optional str or list of str.  When given, only the named
                            trial(s) are loaded and processed.  Pass a single trial
                            name (e.g. "Trial01") or a list.  None → all trials.
        show              : If True, display raw signal quality plot after loading.
    """
    df_ppg_raw, df_events, participant_id = load_ppg(participants_path, trial_filter=trial_filter)
    df_events = clean_events(df_events)

    df_ppg = df_ppg_raw.copy()
    df_ppg = df_ppg.rename(columns={'Internal ADC A13': 'PPG'})

    ts  = df_ppg["Time Stamp"].astype(float)
    rel = ts - ts.iloc[0]
    df_ppg["rel_zero_ref"]  = rel
    df_ppg["abs_zero_ref"]  = rel          # single trial: abs == rel (no offset)
    df_ppg.index            = pd.to_timedelta(rel, unit='ms')
    df_ppg["time_seconds"]  = df_ppg["rel_zero_ref"] / 1000

    dup_mask = df_ppg.index.duplicated(keep='first')
    n_dup    = int(dup_mask.sum())
    if n_dup:
        trial_label = df_ppg["trial"].iloc[0]
        print(f"  {trial_label}: {n_dup} duplicate timestamps removed")
        df_ppg = df_ppg[~dup_mask]

    badsegments             = find_bad_segments(df_ppg)
    df_ppg, fs, badsegments = resample_signal(df_ppg, badsegments, fs_new=250)

    # Replace interpolated values with exact regular-grid times
    rel_ms                 = df_ppg.index.total_seconds() * 1000
    df_ppg["rel_zero_ref"] = rel_ms
    df_ppg["abs_zero_ref"] = rel_ms
    df_ppg["time_seconds"] = rel_ms / 1000

    if show:
        check_signals_ppg(df_ppg, badsegments, df_events)
        plt.show()

    return df_ppg, df_events, fs, badsegments, participant_id
