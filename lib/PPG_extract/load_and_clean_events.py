import re
import pandas as pd
import numpy as np
from pathlib import Path
from lib.config import MAX_TASK_TRIALS, EXCLUDE_PAIRS


def verify_participant(events, participants_path):
    participants_path = Path(participants_path)

    # folder number: JS_01 = 1
    m_folder = re.search(r"(\d+)$", participants_path.name)
    folder_id = int(m_folder.group(1)) if m_folder else None

    # event number: detail of experiment_start, "participant=001" = 1
    detail = events.loc[events["event_label"] == "experiment_start", "detail"]
    m_event = re.search(r"participant\s*=+\s*(\d+)", str(detail.iloc[0])) if not detail.empty else None
    event_num = int(m_event.group(1)) if m_event else None

    match_id = folder_id is not None and event_num is not None and folder_id == event_num
    if event_num is None:
        print("  verify: no 'participant=###' found in experiment_start detail")
    print(f"  verify: folder={participants_path.name} | event participant={event_num} "
          f"-> {'OK' if match_id else 'MISMATCH'}")

    # number of sound stimuli = highest trial number in the event log
    trials = pd.to_numeric(events["trial"], errors="coerce").dropna()
    n_trials = trials.max() if not trials.empty else np.nan
    correct_trials = bool(n_trials == MAX_TASK_TRIALS and trials.nunique() == MAX_TASK_TRIALS)
    print(f"  verify: trials max={n_trials}, distinct={trials.nunique()}, "
          f"expected {MAX_TASK_TRIALS} -> {'OK' if correct_trials else 'MISMATCH'}")

    return match_id, correct_trials


def _paired_intervals(events, start_label, end_label):
    s = np.sort(events.loc[events["event_label"] == start_label, "monotonic_s"].to_numpy(float))
    e = np.sort(events.loc[events["event_label"] == end_label, "monotonic_s"].to_numpy(float))
    if len(s) != len(e):
        raise ValueError(f"Unbalanced markers: {len(s)} {start_label} vs {len(e)} {end_label}")
    return list(zip(s, e))


def _trim(df, t0, t1):
    # keep rows from block start to block end (inclusive); time is NOT collapsed
    t = df["monotonic_s"].astype(float)
    return df.loc[(t >= t0) & (t <= t1)].copy()


def load_and_clean_events(touch_data, participants_path, show=False):
    participants_path = Path(participants_path)
    # load events 
    event_files = sorted(participants_path.glob("event_log_*.csv"))
    if len(event_files) != 1:
        raise ValueError(
            f"Expected exactly one event_log_*.csv in {participants_path}, found {len(event_files)}"
        )
    events = pd.read_csv(event_files[0])

    # touch_data is passed in, already loaded in full_process_single
    if touch_data is None:
        raise ValueError(f"No usable touch data for {participants_path.name}")
    if "monotonic_s" not in touch_data.columns:
        raise ValueError(f"touch data for {participants_path.name} has no monotonic_s column")

    # verify participant id and nb of trials
    match_id, correct_trials = verify_participant(events, participants_path)
    if not match_id:
        raise ValueError(f"Event log does not match participant folder {participants_path.name}")
    if not correct_trials:
        print(f"  WARNING: {participants_path.name} trial count != {MAX_TASK_TRIALS}")

    # trim events from block start until block end
    ev_t = events["monotonic_s"].astype(float)
    t0 = ev_t[events["event_label"] == "block_start"].min()
    t1 = ev_t[events["event_label"] == "block_end"].max()
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("block_start / block_end marker missing")

    # record sound_gap and task_pause intervals as skips; they are KEPT in the data
    # (the silence around each sound is used for plotting)
    intervals = []
    for a, b in EXCLUDE_PAIRS:
        kind = a.removesuffix("_start")
        intervals += [(s, e, kind) for s, e in _paired_intervals(events, a, b)]
    intervals = sorted((max(s, t0), min(e, t1), k) for s, e, k in intervals if e > t0 and s < t1)
    starts = np.array([s for s, _, _ in intervals], dtype=float)
    ends = np.array([e for _, e, _ in intervals], dtype=float)
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError("Overlapping sound_gap / task_pause intervals")

    # trim events and touch data to the task block (no gaps removed)
    df_events = _trim(events, t0, t1)
    df_touch = _trim(touch_data, t0, t1)

    # zero-reference monotonic_s (events, touch and skips share the same clock)
    origin = df_events["monotonic_s"].astype(float).min()
    # rounded to the microsecond so float error cannot push a sample across a boundary
    df_events["monotonic_s"] = (df_events["monotonic_s"] - origin).round(6)
    df_touch["monotonic_s"] = (df_touch["monotonic_s"] - origin).round(6)
    df_events = df_events.reset_index(drop=True)
    df_touch = df_touch.reset_index(drop=True)
    skips = pd.DataFrame({
        "type":     [k for _, _, k in intervals],
        "start":    (starts - origin).round(6),
        "end":      (ends - origin).round(6),
        "duration": ends - starts,
    })

    if show:
        print(f"  Loaded {participants_path.name} | events {len(events)}->{len(df_events)} "
              f"| touch {len(touch_data)}->{len(df_touch)} "
              f"| kept {len(skips)} gaps/pauses ({skips['duration'].sum():.1f}s)")

    # return cleaned events, cleaned touch
    return df_events, df_touch, skips
