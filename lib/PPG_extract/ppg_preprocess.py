import pandas as pd
from lib.config import *


def clean_events(df_events):
    """
    Retain only the event rows needed for analysis: event_types listed in ALLCOND.
    """
    allowed_types = {t[0] for t in ALLCOND}
    return df_events[df_events['event_type'].isin(allowed_types)].reset_index(drop=True)


def assign_trial_condition(df_events):
    """
    Stamp `trial` / `condition` columns onto a single-session event log by
    locating each anchor event's row position (rows are already ordered by
    event_number). One continuous recording covers baseline and 6 stimulus
    blocks:

        trial=0        condition='baseline'       baseline_start .. baseline_end
        trial=1..6     condition=<block label>     block_start .. rest_end
                                                     (or .. post_recovery_end
                                                     for block 6, which has no
                                                     rest_start/rest_end of its
                                                     own — post_recovery IS its
                                                     recovery phase)

    Rows outside all of the above windows are left with trial/condition = NaN;
    `clean_events` only keeps ALLCOND event types, all of which fall inside
    one of these windows, so no valid row is ever left untagged.
    """
    df = df_events.sort_values('event_number').reset_index(drop=True)
    trial     = pd.Series(pd.NA, index=df.index, dtype='object')
    condition = pd.Series(pd.NA, index=df.index, dtype='object')

    def mark(lo_idx, hi_idx, trial_val, cond_val):
        span = (df.index >= lo_idx) & (df.index <= hi_idx)
        trial.loc[span]     = trial_val
        condition.loc[span] = cond_val

    # ── Baseline ────────────────────────────────────────────────────────────
    b_start = df.index[df['event_type'] == 'baseline_start']
    b_end   = df.index[df['event_type'] == 'baseline_end']
    if len(b_start) and len(b_end):
        mark(b_start[0], b_end[0], 0, 'baseline')

    # ── Blocks ──────────────────────────────────────────────────────────────
    # block_start's own row is the sole source of truth for this block's
    # number/label/design — the countdown_start/sound_play_* rows in between
    # reuse the 'trial'/'block' raw columns for unrelated internal indices
    # (e.g. sub-numbering the 12 sounds, or a per-design occurrence counter),
    # which must NOT be mistaken for the session-wide block number.
    for _, row in df[df['event_type'] == 'block_start'].iterrows():
        start_idx  = row.name
        block_num  = int(row['raw_block_index'])
        # Condition = sound label + block design, e.g. 'loud_individu' vs
        # 'loud_quatre_sons' — same sound label, different stimulus design,
        # so they must not collapse into one condition.
        block_cond = f"{row['raw_block_label']}_{row['detail']}"

        block_end_idx = df.index[
            (df['event_type'] == 'block_end') & (df.index > start_idx)
        ]
        if not len(block_end_idx):
            continue
        stop_idx = block_end_idx[0]

        # Recovery-equivalent window right after this block: rest_start/end
        # for blocks 1-5, falling back to post_recovery_start/end for block 6
        # (which has no rest of its own — post_recovery plays that role).
        next_block_idx = df.index[
            (df['event_type'] == 'block_start') & (df.index > stop_idx)
        ]
        rest_end_idx = df.index[
            (df['event_type'] == 'rest_end') & (df.index > stop_idx)
        ]
        if len(rest_end_idx) and (
            not len(next_block_idx) or rest_end_idx[0] < next_block_idx[0]
        ):
            stop_idx = rest_end_idx[0]
        else:
            pr_end_idx = df.index[
                (df['event_type'] == 'post_recovery_end') & (df.index > stop_idx)
            ]
            if len(pr_end_idx) and (
                not len(next_block_idx) or pr_end_idx[0] < next_block_idx[0]
            ):
                stop_idx = pr_end_idx[0]

        mark(start_idx, stop_idx, block_num, block_cond)

    df['trial']     = trial
    df['condition'] = condition
    return df


def merge_time(df_ppg):
    """
    Adds two zero-referenced time columns to df_ppg:

    rel_zero_ref — each trial independently zeroed to its own first sample.
                   Shares the same time base as df_events['time_since_connected_ms']
                   so events can be aligned per-trial.

    abs_zero_ref — continuous timeline across all trials: each trial is zeroed
                   relative to its own first sample, then offset by the cumulative
                   end-time of all preceding trials so the full recording is
                   represented as one unbroken axis.
    """
    df_ppg = df_ppg.copy()
    running_offset = 0
    rel_parts, abs_parts = [], []

    for trial in sorted(df_ppg["trial"].unique()):
        mask = df_ppg["trial"] == trial
        trial_ts = df_ppg.loc[mask, "Time Stamp"]
        rel = trial_ts - trial_ts.iloc[0]
        abs_ = rel + running_offset
        rel_parts.append(rel)
        abs_parts.append(abs_)
        # Advance by one sample period so the next trial's first sample
        # doesn't collide with this trial's last sample in abs_zero_ref
        sample_ms = float(trial_ts.diff().median())
        running_offset = abs_.iloc[-1] + sample_ms

    df_ppg["rel_zero_ref"] = pd.concat(rel_parts)
    df_ppg["abs_zero_ref"] = pd.concat(abs_parts)
    return df_ppg


def zero_reference(df_ppg):
    """
    Returns
    -------
    df_ppg : DataFrame with abs_zero_ref zeroed to the first sample.
             rel_zero_ref is unchanged — it is already connection-relative
             (same origin as time_since_connected_ms and peak_time_s).
    t0     : {'abs_zero_ref': float} — ms offset subtracted from abs_zero_ref.
    """
    df_ppg = df_ppg.copy()
    t0 = {}
    t0["abs_zero_ref"] = float(df_ppg["abs_zero_ref"].iloc[0])
    df_ppg["abs_zero_ref"] = df_ppg["abs_zero_ref"] - t0["abs_zero_ref"]
    return df_ppg, t0



def correct_time(df, df_events):

    if df.columns.shape[0] == 1:
        Time = (df.index[:] - df.index[0]) * 1000

        x = Time.values

        time_idx = pd.to_timedelta(Time, unit='ms')

        fs = 1 / (x[1] - x[0])
        df.index = time_idx

    else:
        Time = (df.iloc[:, 0] - df.iloc[0, 0]) * 1000

        x = Time.values
        time_idx = pd.to_timedelta(Time, unit='ms')

        fs = 1 / (x[1] - x[0]) # sampling rate (freq sample)
        df.index = time_idx

    Time_s = df.index.total_seconds()
    df.insert(1, 'time_seconds', Time_s)

    if df_events is not None:

        event_time = (df_events.iloc[:, 0] - df.iloc[0, 0]) * 1000

        x = event_time.values / 1000
        time_idx = pd.to_timedelta(event_time, unit='ms')

        df_events.index = time_idx

        Time_s = df_events.index.total_seconds()
        df_events.insert(1, 'time_seconds', Time_s)

    return df, df_events, fs


def extract_task_epochs(df_ppg, df_events):
    """
    Crop df_ppg to the event window [first_event, last_event] per trial.

    Uses rel_zero_ref (exact after per-trial resampling) to align with
    df_events['time_since_connected_ms'].  All events in between the first
    and last are treated as markers only, not additional crop boundaries.

    Returns:
        df_epoch     : df_ppg rows within the event window (all trials combined,
                       abs_zero_ref timeline preserved).
        epoch_bounds : {trial: {"rel_lo","rel_hi","abs_lo","abs_hi"}} in ms.
    """
    epoch_bounds = {}
    keep = pd.Series(False, index=df_ppg.index)

    for trial in sorted(df_ppg["trial"].dropna().unique()):
        ev_ms  = df_events.loc[df_events["trial"] == trial, "time_since_connected_ms"]
        t_mask = df_ppg["trial"] == trial

        if ev_ms.empty:
            print(f"  {trial}: no events — keeping full trial")
            keep |= t_mask
            continue

        first_ev = float(ev_ms.min())
        last_ev  = float(ev_ms.max())
        t_start  = float(df_ppg.loc[t_mask, "abs_zero_ref"].iloc[0])

        epoch_bounds[trial] = {
            "rel_lo": first_ev, "rel_hi": last_ev,
            "abs_lo": t_start + first_ev, "abs_hi": t_start + last_ev,
        }

        ep_mask = t_mask & (df_ppg["rel_zero_ref"] >= first_ev) & (df_ppg["rel_zero_ref"] <= last_ev)
        keep   |= ep_mask

        n_kept  = int(ep_mask.sum())
        n_trial = int(t_mask.sum())
        print(f"  {trial}: [{first_ev/1000:.1f}s – {last_ev/1000:.1f}s]  {n_kept}/{n_trial} rows kept")

    return df_ppg.loc[keep], epoch_bounds


def crop_dataset(df_ppg, df_events):
    if df_events is None or len(df_events) == 0:
        return df_ppg

    keep = pd.Series(False, index=df_ppg.index)
    for trial in df_ppg["trial"].unique():
        ev_ts = df_events.loc[df_events["trial"] == trial, "time_since_connected_ms"]
        if ev_ts.empty:
            continue
        ppg_trial = df_ppg["trial"] == trial
        keep |= (
            ppg_trial
            & (df_ppg["rel_zero_ref"] >= ev_ts.iloc[0])
            & (df_ppg["rel_zero_ref"] <= ev_ts.iloc[-1])
        )
    return df_ppg.loc[keep]
