# EDA (electrodermal activity) metric extraction
#
# Mirrors HRV_temp_extract.py's get_temp_metrics: computes summary metrics
# from a preprocess_visualize_gsr results dict, optionally windowed to
# [t_start, t_end] (seconds) via the same masking convention used throughout
# this codebase.
#
# Metric names are prefixed by component (Tonic_/Phasic_) so the origin of
# each metric (skin conductance level vs. skin conductance response) is
# explicit without relying on SCL/SCR jargon.

import numpy as np


def get_eda_metrics(results, t_start=None, t_end=None):
    """
    Compute tonic (SCL) and phasic (SCR) EDA metrics from a
    preprocess_visualize_gsr results dict.

    Parameters
    ----------
    results : dict -- output of preprocess_visualize_gsr (lib.GSR_extract.gsr_preprocess).
    t_start, t_end : float, optional -- window bounds in seconds. If either is
                     None, the whole recording in `results` is used.

    Returns
    -------
    dict with keys:
        Tonic_SCL_mean            -- mean tonic level over the window (uS)
        Tonic_SCL_slope           -- linear-regression slope of tonic over the window (uS/s)
        Phasic_SCR_count          -- number of qualifying SCR peaks in the window
        Phasic_SCR_rate           -- SCR count normalized to responses/minute
        Phasic_SCR_amplitude_mean -- mean amplitude of qualifying SCRs (uS)
        Phasic_SCR_amplitude_sum  -- summed amplitude of qualifying SCRs (uS)
        Phasic_AUC                -- trapezoidal area under the phasic curve (uS.s)
    Empty dict if the window contains fewer than 2 samples.
    """
    time_s = results['time_s']
    tonic  = results['tonic']
    phasic = results['phasic']

    if t_start is not None and t_end is not None:
        lo, hi = t_start, t_end
    else:
        lo, hi = time_s[0], time_s[-1]

    mask = (time_s >= lo) & (time_s <= hi)
    t_win      = time_s[mask]
    tonic_win  = tonic[mask]
    phasic_win = phasic[mask]

    if len(t_win) < 2:
        return {}

    # -- Tonic (SCL) ----------------------------------------------------------
    tonic_scl_mean  = float(np.nanmean(tonic_win))
    tonic_scl_slope = float(np.polyfit(t_win, tonic_win, 1)[0])  # uS/s

    # -- Phasic (SCR) -----------------------------------------------------------
    scr_times     = results['scr_times']
    scr_amplitude = results['scr_amplitude']
    scr_mask      = (scr_times >= lo) & (scr_times <= hi)
    n_scr         = int(scr_mask.sum())
    duration_min  = (hi - lo) / 60.0

    # nanmean/nansum: a peak right at the recording start can have NaN
    # amplitude (NeuroKit2 has no preceding baseline to reference), which
    # would otherwise poison any window containing it.
    amps_win = scr_amplitude[scr_mask]
    with np.errstate(invalid='ignore'):
        phasic_scr_amp_mean = float(np.nanmean(amps_win)) if n_scr > 0 else float('nan')
        phasic_scr_amp_sum  = float(np.nansum(amps_win)) if n_scr > 0 else 0.0
    phasic_scr_count = n_scr
    phasic_scr_rate  = (n_scr / duration_min) if duration_min > 0 else float('nan')
    phasic_auc       = float(np.trapezoid(phasic_win, t_win))

    return {
        'Tonic_SCL_mean':            tonic_scl_mean,
        'Tonic_SCL_slope':           tonic_scl_slope,
        'Phasic_SCR_count':          phasic_scr_count,
        'Phasic_SCR_rate':           phasic_scr_rate,
        'Phasic_SCR_amplitude_mean': phasic_scr_amp_mean,
        'Phasic_SCR_amplitude_sum':  phasic_scr_amp_sum,
        'Phasic_AUC':                phasic_auc,
    }
