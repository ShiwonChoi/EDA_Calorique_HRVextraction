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
# Event markers (lib.config.ALLSTIMCOND) that delineate each phase within a
# stim trial's own recording. Baseline trials (condition == 'baseline') have
# no anticipation/task/recovery split — only baseline_start/baseline_end.
PHASE_EVENTS = {
    'anticipation': ('pre_stimulation_baseline_start', 'pre_stimulation_baseline_end'),
    'task':         ('stimulation_start', 'stimulation_end'),
    'recovery':     ('recovery_start', 'recovery_end'),
}


def phase_windows(df_events_t):
    """
    Map this trial's event_type rows to (start, end) seconds for the
    anticipation / task / recovery phases. Missing markers (e.g. baseline
    trials) yield None for that phase.
    """
    ev = df_events_t.set_index('event_type')['time_since_connected_ms'] / 1000

    windows = {}
    for phase_name, (start_label, end_label) in PHASE_EVENTS.items():
        if start_label in ev.index and end_label in ev.index:
            windows[phase_name] = (float(ev[start_label]), float(ev[end_label]))
        else:
            windows[phase_name] = None
    return windows


def label_bin(bin_center, windows):
    """Phase name whose window contains bin_center, else 'unclassified'."""
    for phase_name, window in windows.items():
        if window is not None and window[0] <= bin_center < window[1]:
            return phase_name
    return 'unclassified'
