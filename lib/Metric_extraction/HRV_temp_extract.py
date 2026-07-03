import numpy as np


def get_temp_metrics(rri_ms, beat_times=None, t_start=None, t_end=None):
    """
    Compute temporal HRV metrics from a cleaned RRI array.

    Parameters
    ----------
    rri_ms      : array-like  — RRI intervals in ms (intervals_clean).
    beat_times  : array-like, optional — time of each beat in seconds (same length as rri_ms).
    t_start     : float, optional — window start in seconds (requires beat_times).
    t_end       : float, optional — window end   in seconds (requires beat_times).

    If beat_times + t_start + t_end are all provided the intervals are filtered
    to [t_start, t_end] before computing metrics.  Otherwise all intervals are used.

    Returns
    -------
    dict with keys: mean_HR, mean_RRI, RMSSD, SDNN
    """
    rri = np.asarray(rri_ms, dtype=float)

    if beat_times is not None and t_start is not None and t_end is not None:
        bt   = np.asarray(beat_times, dtype=float)
        mask = (bt >= t_start) & (bt <= t_end)
        rri  = rri[mask]

    if len(rri) < 2:
        return {}

    mean_rri = float(np.mean(rri))
    mean_hr  = float(60000.0 / mean_rri)
    NNdiff   = np.diff(rri)
    RMSSD    = float(np.sqrt(np.mean(NNdiff ** 2)))
    SDNN     = float(np.std(rri, ddof=1))

    return {
        'mean_HR':  mean_hr,
        'mean_RRI': mean_rri,
        'RMSSD':    RMSSD,
        'SDNN':     SDNN,
    }


# ==========================================================================
# RECORDING-PHASE BOUNDARIES (shared by HRV_freq_bin and HRV_temp_bin)
# ==========================================================================
# Event markers (lib.config.ALLCOND) that delineate each phase within a
# block trial's own recording. Baseline trials have no anticipation/task/
# recovery split — only their own start/end markers. 'recovery' has two
# candidate marker pairs: rest_start/rest_end for blocks 1-5, falling back to
# post_recovery_start/post_recovery_end for block 6, which has no rest of its
# own (assign_trial_condition extends block 6's window through post_recovery
# for exactly this reason).
PHASE_EVENTS = {
    'anticipation': [('block_start', 'countdown_start')],
    'task':         [('sound_play_start', 'sound_play_end')],
    'recovery':     [('rest_start', 'rest_end'), ('post_recovery_start', 'post_recovery_end')],
}


def phase_windows(df_events_t):
    """
    Map this trial's event_type rows to (start, end) seconds for the
    anticipation / task / recovery phases. Missing markers (e.g. baseline
    trials) yield None for that phase. Aggregates by min/max rather than a
    unique index lookup since a quatre_sons block has up to 12
    sound_play_start/end rows — the task window spans the first
    sound_play_start to the last sound_play_end.
    """
    ev = df_events_t.groupby('event_type')['time_since_connected_ms'].agg(['min', 'max']) / 1000

    windows = {}
    for phase_name, candidates in PHASE_EVENTS.items():
        windows[phase_name] = None
        for start_label, end_label in candidates:
            if start_label in ev.index and end_label in ev.index:
                windows[phase_name] = (float(ev.loc[start_label, 'min']), float(ev.loc[end_label, 'max']))
                break
    return windows


def label_bin(bin_center, windows):
    """Phase name whose window contains bin_center, else 'unclassified'."""
    for phase_name, window in windows.items():
        if window is not None and window[0] <= bin_center < window[1]:
            return phase_name
    return 'unclassified'
