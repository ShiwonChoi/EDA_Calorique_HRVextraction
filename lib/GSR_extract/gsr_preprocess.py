# GSR/EDA preprocessing
#
# Mirrors Metric_extraction/RRI_preprocess.py's structure (artifact
# detection/removal -> preprocess_visualize orchestrator -> plot) for the
# electrodermal-activity channel: unit conversion, artifact handling,
# tonic/phasic decomposition, and SCR peak detection.

import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from lib.config import (
    GSR_PLAUSIBLE_RANGE_US, GSR_MAX_SLOPE_US_PER_S,
    GSR_DECOMPOSITION_METHOD, SCR_AMPLITUDE_MIN_US,
)


def resistance_to_conductance(gsr_kohm):
    """
    Convert Shimmer GSR skin resistance (kOhm) to skin conductance (microSiemens).

    Conductance = 1000 / Resistance(kOhm). Conductance is the preferred EDA
    analysis domain (Dawson, Schell & Filion, 2016; Boucsein, 2012) since SCR
    amplitude/morphology is approximately linear in conductance but
    compressed/expanded nonlinearly in resistance.
    """
    gsr_kohm = np.asarray(gsr_kohm, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        eda_uS = 1000.0 / gsr_kohm
    return eda_uS


def detect_artifacts_gsr(eda_uS, fs,
                          plausible_range=GSR_PLAUSIBLE_RANGE_US,
                          max_slope=GSR_MAX_SLOPE_US_PER_S):
    """
    Flag artifact samples in a conductance signal (microSiemens).

    Two independent criteria, mirroring RRI_preprocess.detect_artifacts's
    two-tier pattern:
      1. Physiological bounds: outside `plausible_range` (default 0.05-60 uS,
         Kleckner et al. 2018) -- catches non-finite/negative/implausible
         values from Shimmer auto-range switching.
      2. Max slope: |d(conductance)/dt| > max_slope (default 10 uS/s,
         Kleckner et al. 2018) -- catches abrupt jumps at range-switch
         boundaries.

    Returns
    -------
    artifact_mask : np.ndarray (bool) -- True = artifact.
    """
    eda_uS = np.asarray(eda_uS, dtype=float)
    lo, hi = plausible_range

    artifact_mask = ~np.isfinite(eda_uS) | (eda_uS < lo) | (eda_uS > hi)
    print(f"    Physio range filter ({lo}-{hi} uS): {artifact_mask.sum()} artifacts flagged")

    slope = np.zeros_like(eda_uS)
    slope[1:] = np.diff(eda_uS) * fs
    slope_mask = np.abs(slope) > max_slope
    n_new = int((slope_mask & ~artifact_mask).sum())
    artifact_mask |= slope_mask
    print(f"    Max-slope filter ({max_slope} uS/s): {n_new} additional artifacts flagged")

    return artifact_mask


def remove_artifacts_gsr(eda_uS, artifact_mask):
    """
    Linearly interpolate over artifact samples in a conductance signal.

    Mirrors RRI_preprocess.remove_artifacts's 'interpolate' method, adapted
    for a regularly-sampled signal (interpolates in sample-index space).
    """
    eda_uS = np.asarray(eda_uS, dtype=float).copy()
    idx   = np.arange(len(eda_uS))
    valid = ~artifact_mask

    if valid.sum() < 2:
        return eda_uS

    eda_uS[artifact_mask] = np.interp(idx[artifact_mask], idx[valid], eda_uS[valid])
    return eda_uS


def masked_sample_counts(results, t_start, t_end):
    """
    Count GSR samples within [t_start, t_end]: total (raw) samples and
    non-artifact (clean) samples, mirroring the "<n_clean> / <n_raw>"
    sample_size convention used for RRI beats elsewhere in this pipeline.

    Returns
    -------
    n_clean, n_raw : int
    """
    mask  = (results['time_s'] >= t_start) & (results['time_s'] <= t_end)
    n_raw   = int(mask.sum())
    n_clean = int((mask & ~results['artifact_mask']).sum())
    return n_clean, n_raw


def decompose_eda(eda_clean, fs, method=GSR_DECOMPOSITION_METHOD):
    """
    Split a cleaned EDA signal into tonic (SCL) and phasic (SCR) components.

    Thin wrapper around nk.eda_phasic. method='highpass' (default,
    Butterworth high-pass/low-pass split -- Biopac/AcqKnowledge convention,
    no independent peer-reviewed origin) or 'cvxeda' (Greco et al. 2016,
    convex optimization, requires the cvxopt package) are the two primary
    supported options here; 'sparse' (Hernando-Gallego et al. 2018) is
    available in NeuroKit2 but flagged experimental by NeuroKit2 itself.

    Returns
    -------
    tonic, phasic : np.ndarray
    """
    decomposed = nk.eda_phasic(eda_clean, sampling_rate=fs, method=method)
    return decomposed['EDA_Tonic'].values, decomposed['EDA_Phasic'].values


def detect_scr_peaks(phasic, fs, amplitude_min=SCR_AMPLITUDE_MIN_US):
    """
    Detect skin conductance response (SCR) peaks in a phasic EDA signal.

    Thin wrapper around nk.eda_peaks.

    Returns
    -------
    info : dict -- SCR_Onsets/SCR_Peaks (sample indices into `phasic`),
           SCR_Amplitude (uS), SCR_RiseTime, SCR_Recovery, SCR_RecoveryTime,
           sampling_rate.
    """
    _, info = nk.eda_peaks(phasic, sampling_rate=fs, amplitude_min=amplitude_min)
    return info


def preprocess_visualize_gsr(time_s, gsr_kohm, fs,
                              method=GSR_DECOMPOSITION_METHOD,
                              amplitude_min=SCR_AMPLITUDE_MIN_US,
                              verbose=True):
    """
    Full GSR/EDA preprocessing pipeline: unit conversion, artifact detection
    and interpolation, NeuroKit2 cleaning, tonic/phasic decomposition, and
    SCR peak detection.

    Parameters
    ----------
    time_s   : array-like -- time axis (seconds, shimmer-connected), same
               length as gsr_kohm.
    gsr_kohm : array-like -- raw Shimmer GSR channel (kOhm).
    fs       : float -- sampling rate (Hz).

    Returns
    -------
    results : dict
        'time_s'        : time axis (seconds)
        'eda_raw'       : raw conductance (uS), before artifact removal
        'artifact_mask' : bool mask, True = artifact
        'eda_clean'     : conductance after artifact interpolation + nk.eda_clean
        'tonic'         : tonic (SCL) component (uS)
        'phasic'        : phasic (SCR) component (uS)
        'scr_times'     : time (s) of each detected SCR peak
        'scr_amplitude' : amplitude (uS) of each detected SCR peak
        'fs'            : sampling rate used
        'method'        : decomposition method used
        'amplitude_min' : SCR amplitude threshold used
    """
    time_s = np.asarray(time_s, dtype=float)

    if verbose:
        print(f"\n--Raw GSR: {len(gsr_kohm)} samples at {fs} Hz--")

    eda_raw = resistance_to_conductance(gsr_kohm)

    artifact_mask = detect_artifacts_gsr(eda_raw, fs)
    n_artifacts = int(artifact_mask.sum())
    if verbose:
        print(f"    Artifacts detected: {n_artifacts} ({100 * n_artifacts / len(eda_raw):.1f}%)")

    eda_interp = remove_artifacts_gsr(eda_raw, artifact_mask)
    eda_clean  = nk.eda_clean(eda_interp, sampling_rate=fs)

    tonic, phasic = decompose_eda(eda_clean, fs, method=method)
    if verbose:
        print(f"    Decomposed via '{method}': tonic mean={np.nanmean(tonic):.3f} uS, "
              f"phasic std={np.nanstd(phasic):.3f} uS")

    scr_info     = detect_scr_peaks(phasic, fs, amplitude_min=amplitude_min)
    scr_peak_idx = np.asarray(scr_info['SCR_Peaks'], dtype=int)
    scr_times     = time_s[scr_peak_idx] if scr_peak_idx.size else np.array([])
    scr_amplitude = (np.asarray(scr_info['SCR_Amplitude'], dtype=float)
                     if scr_peak_idx.size else np.array([]))
    if verbose:
        print(f"    SCR peaks detected: {len(scr_times)} (amplitude_min={amplitude_min} uS)")

    return {
        'time_s':        time_s,
        'eda_raw':       eda_raw,
        'artifact_mask': artifact_mask,
        'eda_clean':     eda_clean,
        'tonic':         tonic,
        'phasic':        phasic,
        'scr_times':     scr_times,
        'scr_amplitude': scr_amplitude,
        'fs':            fs,
        'method':        method,
        'amplitude_min': amplitude_min,
    }


def plot_preprocessing_steps_gsr(results, participant_id=None, df_events=None, show=True):
    """
    Visualize GSR preprocessing: raw conductance with artifacts marked,
    cleaned signal, and tonic/phasic decomposition with detected SCR peaks.

    If df_events is provided, vertical lines are drawn at each event time
    (start events in black, end events in red) -- mirrors
    RRI_preprocess.plot_preprocessing_steps' event-marker convention.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    t = results['time_s']

    # Step 1 -- raw conductance with artifacts marked
    axes[0].plot(t, results['eda_raw'], color='#1f77b4', linewidth=1, alpha=0.8)
    bad = results['eda_raw'].copy()
    bad[~results['artifact_mask']] = np.nan
    axes[0].scatter(t, bad, color='red', s=10, label='Detected artifacts', zorder=5)
    axes[0].set_ylabel('Conductance (uS)', fontsize=10)
    axes[0].set_title('Step 1: Raw Conductance (red = artifacts)', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Step 2 -- cleaned signal
    axes[1].plot(t, results['eda_clean'], color='#ff7f0e', linewidth=1, alpha=0.9)
    axes[1].set_ylabel('Conductance (uS)', fontsize=10)
    axes[1].set_title('Step 2: After Artifact Interpolation + Cleaning', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Step 3 -- tonic/phasic decomposition with SCR peaks
    axes[2].plot(t, results['tonic'], color='#2ca02c', linewidth=1.2, label='Tonic (SCL)')
    axes[2].plot(t, results['phasic'], color='#9467bd', linewidth=0.8, alpha=0.8, label='Phasic (SCR)')
    if len(results['scr_times']):
        peak_vals = np.interp(results['scr_times'], t, results['phasic'])
        axes[2].scatter(results['scr_times'], peak_vals, color='black', s=20,
                         zorder=5, label='SCR peaks')
    axes[2].set_ylabel('Conductance (uS)', fontsize=10)
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].set_title(f"Step 3: Tonic/Phasic Decomposition ({results['method']})",
                       fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    if df_events is not None:
        for _, row in df_events.iterrows():
            t_s = row['time_since_connected_ms'] / 1000
            color = 'red' if str(row['event_type']).endswith('end') else 'black'
            for ax in axes:
                ax.axvline(x=t_s, color=color, linestyle='--', linewidth=0.7, alpha=0.5)

    if participant_id is not None:
        fig.suptitle(f"GSR/EDA preprocessing -- {participant_id}", fontsize=13, fontweight='bold', y=1.00)

    plt.tight_layout()

    if show:
        plt.show()
        plt.close(fig)

    return fig
