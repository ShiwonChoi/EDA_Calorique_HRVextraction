# Tachogram preprocessing pipeline 

"""
Preprocessing utilities for HRV/PRV interval data.

This module handles artifact detection, detrending, resampling,
and other preprocessing steps for beat interval analysis.

Evidence-based approach following:
- Task Force 1996 (standards)
- Clifford & Tarassenko 2005 (beat replacement effects)
- Mejía-Mejía 2022 (PRV optimal parameters)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


def detect_artifacts(
    intervals: np.ndarray,
    use_physio: bool = True,
    use_stat: bool = False,
    low_pass_stat: bool = True,
    min_interval: float = 300,
    max_interval: float = 2000,
    local_outlier_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally applies up to three independent artifact criteria:

    1. Physiological bounds (use_physio): flag intervals outside [min_interval, max_interval].
    2. Statistical / local MAD (use_stat): flag intervals deviating > local_outlier_threshold
       * MAD from a rolling median (window = 5% of signal, min 5 beats), i.e. rejects
       intervals below (rolling_median - threshold*MAD) or above (rolling_median + threshold*MAD).
    3. Low-pass statistical (low_pass_stat): one-sided version of criterion 2 — only rejects
       intervals above (rolling_median + threshold*MAD); intervals below the lower bound are
       kept.

    Returns
    -------
    artifact_mask : np.ndarray
        Boolean mask (True = artifact).
    artifact_indices : np.ndarray
        Indices where artifacts were detected.
    """
    artifact_mask = np.zeros(len(intervals), dtype=bool)

    if use_physio:
        artifact_mask |= (intervals < min_interval) | (intervals > max_interval)
        print(f"    Physio filter (300–2000 ms): {artifact_mask.sum()} artifacts flagged")

    if use_stat or low_pass_stat:
        window_size = max(5, int(len(intervals) * 0.05))
        s = pd.Series(intervals)
        rolling_median = s.rolling(window_size, center=True, min_periods=1).median().values
        mad = np.median(np.abs(intervals - rolling_median))
        deviation = intervals - rolling_median

        if use_stat:
            stat_mask = np.abs(deviation) > local_outlier_threshold * mad
            n_new = int((stat_mask & ~artifact_mask).sum())
            artifact_mask |= stat_mask
            print(f"    MAD filter (threshold={local_outlier_threshold}): {n_new} additional artifacts flagged")

        if low_pass_stat:
            low_pass_mask = deviation > local_outlier_threshold * mad
            n_new = int((low_pass_mask & ~artifact_mask).sum())
            artifact_mask |= low_pass_mask
            print(f"    Low-pass MAD filter (threshold={local_outlier_threshold}, upper bound only): {n_new} additional artifacts flagged")

    artifact_indices = np.where(artifact_mask)[0]
    return artifact_mask, artifact_indices


def remove_artifacts(
    intervals: np.ndarray,
    artifact_mask: np.ndarray,
    method: str = 'remove'
    ) -> np.ndarray:
    """
    Remove or interpolate artifact intervals.
    
    Returns
    -------
    processed_intervals : np.ndarray
        Processed interval data
    """
    if method == 'remove':
        return intervals[~artifact_mask]
    
    elif method == 'mean':
        result = intervals.copy()
        window_size = max(3, int(len(intervals) * 0.05))
        for i in np.where(artifact_mask)[0]:
            start = max(0, i - window_size//2)
            end = min(len(intervals), i + window_size//2)
            # Calculate mean of non-artifact neighbors
            neighbors = ~artifact_mask[start:end]
            if neighbors.any():
                result[i] = np.mean(intervals[start:end][neighbors])
        return result
    
    elif method == 'interpolate':
        result = intervals.copy()
        artifact_indices = np.where(artifact_mask)[0]
        
        if len(artifact_indices) == 0:
            return result
        
        # Find valid indices for interpolation
        all_indices = np.arange(len(intervals))
        valid_indices = all_indices[~artifact_mask]
        
        if len(valid_indices) < 2:
            return result  # Can't interpolate with < 2 points
        
        # Linear interpolation for artifacts
        for idx in artifact_indices:
            # Find nearest valid neighbors
            before = valid_indices[valid_indices < idx]
            after = valid_indices[valid_indices > idx]
            
            if len(before) > 0 and len(after) > 0:
                idx_before = before[-1]
                idx_after = after[0]
                # Linear interpolation
                weight = (idx - idx_before) / (idx_after - idx_before)
                result[idx] = (1 - weight) * result[idx_before] + weight * result[idx_after]
        
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detrend(
    intervals: np.ndarray,
    method: str = 'polynomial',
    order: int = 1) -> np.ndarray:
    """
    Remove slow trends from interval data.
    
    Returns
    -------
    detrended : np.ndarray
        Detrended interval data
    """
    if method == 'mean':
        return intervals - np.mean(intervals)
    
    elif method == 'polynomial':
        x = np.arange(len(intervals))
        coeffs = np.polyfit(x, intervals, order)
        trend = np.polyval(coeffs, x)
        return intervals - trend
    
    else:
        raise ValueError(f"Unknown detrending method: {method}")


def resample_tachogram(
    beat_times: np.ndarray,
    intervals: np.ndarray,
    target_rate: float = 4.0,
    method: str = 'cubic_spline') -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample tachogram to uniform rate.
    
    Returns
    -------
    t_uniform : np.ndarray
        Uniformly sampled time vector
    intervals_uniform : np.ndarray
        Uniformly resampled intervals
    
    Notes
    -----
    Input intervals are assumed to be time between successive beats.
    Output intervals_uniform are RR/PP values at uniform time points.
    
    Cubic spline recommended because:
    - Smooth interpolation without oscillations
    - Better preserves physiological characteristics
    - Minimizes spectral artifacts
    """
    if len(beat_times) < 3:
        raise ValueError("Need at least 3 beat times for resampling")
    
    # beat_times[i] is the start time of intervals[i] — same length, use directly
    interval_times = beat_times
    
    # Create uniform time vector
    start_time = beat_times[0]
    end_time = beat_times[-1]
    n_samples = int((end_time - start_time) * target_rate) + 1
    t_uniform = np.linspace(start_time, end_time, n_samples)
    
    if method == 'cubic_spline':
        # Cubic spline interpolation
        cs = CubicSpline(interval_times, intervals, bc_type='natural')
        intervals_uniform = cs(t_uniform)
        # Ensure physiologically plausible values
        intervals_uniform = np.clip(intervals_uniform, 300, 2000)
    
    elif method == 'linear':
        # Linear interpolation (simpler but less smooth)
        intervals_uniform = np.interp(t_uniform, interval_times, intervals)
    
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    return t_uniform, intervals_uniform


def preprocess_pipeline(
    beat_times: np.ndarray,
    intervals: np.ndarray,
    use_physio: bool = True,
    use_stat: bool = False,
    remove_artifacts_method: str = 'interpolate',
    detrend_method: str = 'polynomial',
    detrend_order: int = 1,
    resample_rate: float = 4.0,
    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for HRV/PRV analysis.
    
    Applies standard sequence:
    1. Artifact detection and handling
    2. Detrending
    3. Uniform resampling
    
    Parameters
    ----------
    beat_times : np.ndarray
        Time of beat detections (seconds)
    intervals : np.ndarray
        RR/PP intervals (milliseconds)
    remove_artifacts_method : str, optional
        Artifact removal method (default: 'interpolate')
    detrend_method : str, optional
        Detrending method (default: 'polynomial')
    detrend_order : int, optional
        Detrending polynomial order (default: 1)
    resample_rate : float, optional
        Target resampling rate in Hz (default: 4.0)
    verbose : bool, optional
        Print processing information (default: True)
    
    Returns
    -------
    t_processed : np.ndarray
        Processed time vector (uniform)
    intervals_processed : np.ndarray
        Processed intervals (uniform resampling)
    """
    if verbose:
        print(f"Raw intervals: {len(intervals)} samples")
    
    # Step 1: Detect and handle artifacts
    artifact_mask, artifact_indices = detect_artifacts(intervals, use_physio=use_physio, use_stat=use_stat)
    n_artifacts = np.sum(artifact_mask)
    if verbose:
        print(f"Artifacts detected: {n_artifacts} ({100*n_artifacts/len(intervals):.1f}%)")

    intervals_clean = remove_artifacts(intervals, artifact_mask, method=remove_artifacts_method)
    # Only filter beat_times when intervals are physically removed; interpolate/mean keep full length
    if remove_artifacts_method == 'remove':
        beat_times_clean = beat_times[~artifact_mask]
    else:
        beat_times_clean = beat_times
    
    if verbose:
        print(f"After artifact removal: {len(intervals_clean)} samples")
    
    # Step 2: Detrend
    intervals_detrended = detrend(intervals_clean, method=detrend_method, order=detrend_order)
    if verbose:
        print(f"Detrending applied: {detrend_method} (order={detrend_order})")
    
    # Step 3: Resample to uniform rate
    t_uniform, intervals_uniform = resample_tachogram(
        beat_times_clean,
        intervals_detrended,
        target_rate=resample_rate
    )
    
    if verbose:
        print(f"Resampled to {resample_rate} Hz: {len(intervals_uniform)} samples")
        print(f"Duration: {(t_uniform[-1] - t_uniform[0]):.1f} seconds")
    
    return t_uniform, intervals_uniform


def validate_intervals(intervals: np.ndarray) -> dict:
    """
    Calculate summary statistics for interval data quality check.
    
    Returns
    -------
    stats : dict
        Dictionary with quality metrics:
        - 'mean_interval': Mean RR/PP interval
        - 'mean_hr': Mean heart rate (bpm)
        - 'std_interval': Standard deviation
        - 'rmssd': Root mean square of successive differences
        - 'range': Min-max range
        - 'pnn50': Percentage of intervals differing by >50ms
    """
    intervals = np.asarray(intervals)
    
    # Basic statistics
    mean_interval = np.mean(intervals)
    mean_hr = abs(60000 / mean_interval)
    std_interval = np.std(intervals)
    
    # Successive differences
    diff = np.diff(intervals)
    rmssd = np.sqrt(np.mean(diff**2))
    
    # NN50 / pNN50 metric
    nn50 = np.sum(np.abs(diff) > 50)
    pnn50 = 100 * nn50 / len(diff)
    
    return {
        'mean_interval_ms': mean_interval,
        'mean_hr_bpm': mean_hr,
        'std_interval_ms': std_interval,
        'rmssd_ms': rmssd,
        'min_interval_ms': np.min(intervals),
        'max_interval_ms': np.max(intervals),
        'pnn50_percent': pnn50,
        'n_samples': len(intervals)
    }


def preprocess_visualize(
    beat_times: np.ndarray,
    intervals: np.ndarray,
    use_physio: bool = True,
    use_stat: bool = False,
    use_low_pass_stat: bool = True,
    remove_artifacts_method: str = 'remove',
    detrend_method: str = 'polynomial',
    detrend_order: int = 1,
    resample_rate: float = 4.0,
    highpass_cutoff: float = 0.035,
    verbose: bool = True
) -> dict:
    """
    Complete preprocessing pipeline for HRV/PRV analysis.
    
    Applies standard sequence:
    1. Artifact detection and removal
    2. Resampling to uniform rate
    3. High-pass filtering (detrending)
    
    Returns
    -------
    results : dict
        Dictionary containing all intermediate and final results:
        - 'beat_times_raw': original beat times
        - 'intervals_raw': original intervals
        - 'artifact_mask': boolean mask (True = artifact)
        - 'n_artifacts': number of artifacts detected
        - 'beat_times_clean': beat times after artifact removal
        - 'intervals_clean': intervals after artifact removal
        - 't_resampled': uniform time vector
        - 'intervals_resampled': resampled intervals (before filtering)
        - 'intervals_filtered': final filtered intervals
        - 'fs_resample': resampling rate used
        - 'highpass_cutoff': filter cutoff used
    """
    from scipy.signal import butter, filtfilt
    
    if verbose:
        print(f"\n--Raw intervals: {len(intervals)} samples--")
    
    # Store raw data
    beat_times_raw = beat_times.copy()
    intervals_raw = intervals.copy()
    
    # -------------------------------------------------------------------------
    # Step 1: Detect and remove artifacts
    # -------------------------------------------------------------------------
    artifact_mask, artifact_indices = detect_artifacts(intervals_raw, use_physio=use_physio, use_stat=use_stat, low_pass_stat=use_low_pass_stat)
    n_artifacts = np.sum(artifact_mask)

    if verbose:
        print(f"    Artifacts detected: {n_artifacts} ({100*n_artifacts/len(intervals_raw):.1f}%)")
    
    intervals_clean = remove_artifacts(intervals_raw, artifact_mask, method=remove_artifacts_method)
    
    if remove_artifacts_method == 'remove':
        beat_times_clean = beat_times_raw[~artifact_mask]
    else:
        beat_times_clean = beat_times_raw.copy()
    
    if verbose:
        print(f"    After artifact removal: {len(intervals_clean)} samples")
    
    # -------------------------------------------------------------------------
    # Step 2: Resample to uniform rate
    # -------------------------------------------------------------------------
    t_resampled, intervals_resampled = resample_tachogram(
        beat_times_clean,
        intervals_clean,
        target_rate=resample_rate
    )
    
    if verbose:
        print(f"    Resampled to {resample_rate} Hz: {len(intervals_resampled)} samples")
        print(f"    Duration: {(t_resampled[-1] - t_resampled[0]):.1f} seconds")
    
    # -------------------------------------------------------------------------
    # Step 3: High-pass filter (detrending)
    # -------------------------------------------------------------------------
    b, a = butter(4, highpass_cutoff / (resample_rate / 2), btype="high")
    intervals_filtered = filtfilt(b, a, intervals_resampled)
    
    if verbose:
        print(f"    High-pass filtered at {highpass_cutoff} Hz")
    
    # -------------------------------------------------------------------------
    # Return all intermediate results
    # -------------------------------------------------------------------------
    results = {
        # Raw data
        'beat_times_raw': beat_times_raw,
        'intervals_raw': intervals_raw,
        
        # Step 1: Artifact detection
        'artifact_mask': artifact_mask,
        'n_artifacts': n_artifacts,
        
        # Step 2: After artifact removal
        'beat_times_clean': beat_times_clean,
        'intervals_clean': intervals_clean,
        
        # Step 3: After resampling
        't_resampled': t_resampled,
        'intervals_resampled': intervals_resampled,
        
        # Step 4: After filtering (final output)
        'intervals_filtered': intervals_filtered,
        
        # Parameters used
        'fs_resample': resample_rate,
        'highpass_cutoff': highpass_cutoff,
    }
    
    return results


def plot_preprocessing_steps(results, participant_id=None, df_events=None, trial=None, show=True):
    """
    Visualize the four preprocessing steps for an RRI signal: raw with
    artifacts marked, after artifact removal, after resampling, and after
    high-pass filtering.

    If df_events and trial are provided, vertical lines are drawn on all axes
    at each event time (start events in black, end events in red).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=False)

    # Step 1 — raw RRI with artifacts marked
    # (time series of successive PP/RR intervals plotted against time (not uniformly sampled) - inverse of heart rate
    axes[0].plot(results['beat_times_raw'], results['intervals_raw'],
                 'o-', markersize=3, linewidth=1, color='#1f77b4', alpha=0.7)
    axes[0].scatter(results['beat_times_raw'][results['artifact_mask']],
                    results['intervals_raw'][results['artifact_mask']],
                    color='red', s=100, marker='x', linewidth=3,
                    label='Detected artifacts', zorder=5)
    axes[0].set_ylabel("RR Interval (ms)", fontsize=10)
    axes[0].set_title("Step 1: Raw RR Intervals (red = artifacts)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Step 2 — after artifact removal
    axes[1].plot(results['beat_times_clean'], results['intervals_clean'],
                 "o-", markersize=2, linewidth=0.8, color="#ff7f0e", alpha=0.8)
    axes[1].set_ylabel("RR Interval (ms)", fontsize=10)
    axes[1].set_title("Step 2: After Artifact Removal",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Step 3 — after resampling
    axes[2].plot(results['t_resampled'], results['intervals_resampled'],
                 linewidth=1, color="#2ca02c", alpha=0.8)
    axes[2].set_ylabel("RR Interval (ms)", fontsize=10)
    axes[2].set_title(f"Step 3: After Resampling at {results['fs_resample']} Hz",
                      fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # Step 4 — after high-pass filter
    axes[3].plot(results['t_resampled'], results['intervals_filtered'],
                 linewidth=1, color="#d62728", alpha=0.8)
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[3].set_xlabel("Time (s)", fontsize=10)
    axes[3].set_ylabel("RR Interval (ms)", fontsize=10)
    axes[3].set_title(f"Step 4: After High-Pass Filter at "
                      f"{results['highpass_cutoff']} Hz",
                      fontsize=11, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    # Event markers
    if df_events is not None and trial is not None:
        ev = df_events[df_events["trial"] == trial]
        for _, row in ev.iterrows():
            t_s = row["time_since_connected_ms"] / 1000
            color = "red" if str(row["event_type"]).endswith("end") else "black"
            for ax in axes:
                ax.axvline(x=t_s, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
            axes[0].text(t_s, axes[0].get_ylim()[1], row["event_type"],
                         rotation=90, va="top", ha="right", fontsize=7, color=color)

    if participant_id is not None:
        fig.suptitle(f"Preprocessing steps — {participant_id}",
                     fontsize=13, fontweight="bold", y=1.00)

    plt.tight_layout()

    if show:
        plt.show()
        #plt.close(fig)

    return fig