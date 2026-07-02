"""
Wavelet.py
==========
Continuous Wavelet Transform (CWT) analysis for HRV / PRV.

Pipeline architecture (Option B — split compute/task):
    run_cwt_compute()   — called ONCE per participant
        → compute_cwt_power (expensive CWT convolution)

    run_cwt_task()      — called ONCE PER TASK (noise / arith)
        → per_frequency_correction (cheap array arithmetic)
        → extract_band_power_cwt (band integration)
        → band_results_to_df (dict → long-format DataFrame)

Plotting functions (On raw uncorrected CWT):
    plot_cwt_scalogram()
    plot_cwt_band_power()
"""

from pathlib import Path
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lib.config import HRV_BANDS


# numpy >=2.0 renamed np.trapz -> np.trapezoid
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ==========================================================================
# HRV BAND DEFINITIONS (Task Force 1996)
# ==========================================================================
# Default frequency grid — log-spaced for finer resolution in LF/HF
DEFAULT_FREQ_MIN = 0.02
DEFAULT_FREQ_MAX = 0.50
DEFAULT_N_FREQS  = 100

# Adaptive Morlet n_cycles ramp
DEFAULT_N_CYCLES_LOW  = 5
DEFAULT_N_CYCLES_HIGH = 15

# Default reflection-padding length (seconds per side).
# Sized to suppress COI at DEFAULT_FREQ_MIN:
#   max_half_width = 3 * N_CYCLES_LOW / (2π * FREQ_MIN) ≈ 120s for 0.02 Hz
DEFAULT_PAD_SECONDS = 120


# ==========================================================================
# HELPER: max wavelet half-width
# ==========================================================================
def max_wavelet_half_width_seconds(freq_min=None, n_cycles_low=None):
    """
    Truncation half-width (seconds) of the longest wavelet in the CWT grid,
    assuming ±3σ truncation. Used to size reflection padding.
    """
    if freq_min is None:
        freq_min = DEFAULT_FREQ_MIN
    if n_cycles_low is None:
        n_cycles_low = DEFAULT_N_CYCLES_LOW
    return (3 * n_cycles_low) / (2 * np.pi * freq_min)


# ==========================================================================
# CORE CWT COMPUTATION
# ==========================================================================
def adaptive_cwt(signal, frequencies, fs, n_cycles=6):
    """
    Complex Morlet CWT with per-frequency n_cycles control.

    Returns
    -------
    coef : 2-D complex array, shape (n_freqs, n_samples)
        Complex CWT coefficients. Power = |coef|².
    """
    if np.isscalar(n_cycles):
        n_cycles_arr = np.full(len(frequencies), float(n_cycles))
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)

    coef = np.zeros((len(frequencies), len(signal)), dtype=complex)

    for i, (freq, n_cyc) in enumerate(zip(frequencies, n_cycles_arr)):
        s = n_cyc * fs / (2 * np.pi * freq)             # gaussian std (samples)
        M = min(int(2 * 3 * s) + 1, len(signal) - 1)    # wavelet length: ±3 sigma
        M = max(M, 3)
        t = (np.arange(M) - M // 2) / s
        wavelet = (np.exp(1j * n_cyc * t) * np.exp(-0.5 * t ** 2)
                   / np.sqrt(np.sqrt(np.pi) * s))
        coef[i] = np.convolve(signal, wavelet, mode="same")

    return coef


def compute_cwt_power(intervals_filtered, t_resampled, fs,
                      freq_min=DEFAULT_FREQ_MIN,
                      freq_max=DEFAULT_FREQ_MAX,
                      n_freqs=DEFAULT_N_FREQS,
                      n_cycles_low=DEFAULT_N_CYCLES_LOW,
                      n_cycles_high=DEFAULT_N_CYCLES_HIGH,
                      pad_seconds=DEFAULT_PAD_SECONDS,
                      verbose=True):
    """
    Compute CWT power matrix from a preprocessed RRI signal.

    The signal is reflection-padded at both ends before CWT to suppress
    cone-of-influence edge artifacts. Padding is cropped from the output.

    Returns
    -------
    dict with keys:
        'frequencies' : array (n_freqs,)
        'times'       : array (n_samples,) — matches t_resampled
        'power'       : array (n_freqs, n_samples) — |coef|²
        'coef'        : complex CWT coefficients
        'fs'          : sampling rate
        'n_cycles'    : array (n_freqs,)
        'pad_seconds' : padding applied
    """
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = np.linspace(n_cycles_low, n_cycles_high, n_freqs)

    # --- Reflection padding ---
    pad_samples = int(pad_seconds * fs)

    if pad_samples > 0:
        max_avail = len(intervals_filtered) - 1
        if pad_samples > max_avail:
            if verbose:
                print(f"  Reducing pad: requested {pad_seconds:.0f}s "
                      f"but signal only allows {max_avail / fs:.1f}s")
            pad_samples = max_avail

        pad_samples = int(pad_samples)
        signal_padded = np.pad(intervals_filtered,
                               pad_width=pad_samples,
                               mode='reflect')
        if verbose:
            print(f"  Reflection-padded {pad_samples / fs:.0f}s "
                  f"({pad_samples} samples) per side")
    else:
        signal_padded = intervals_filtered
        pad_samples = 0

    # --- CWT on padded signal ---
    if verbose:
        print(f"  Computing CWT: {n_freqs} frequencies, "
              f"{len(signal_padded)} samples (incl. padding)...")

    coef_padded = adaptive_cwt(signal_padded, frequencies, fs, n_cycles=n_cycles)
    power_padded = np.abs(coef_padded) ** 2

    # --- Crop padding ---
    if pad_samples > 0:
        coef  = coef_padded[:, pad_samples:-pad_samples]
        power = power_padded[:, pad_samples:-pad_samples]
    else:
        coef  = coef_padded
        power = power_padded

    if verbose:
        print(f"    Frequencies: {frequencies[0]:.3f}-{frequencies[-1]:.2f} Hz")
        print(f"    Power shape: {power.shape}")

    return {
        'frequencies': frequencies,
        'times':       t_resampled,
        'power':       power,
        'coef':        coef,
        'fs':          fs,
        'n_cycles':    n_cycles,
        'pad_seconds': pad_seconds,
    }


# ==========================================================================
# PER-FREQUENCY BASELINE CORRECTION
# ==========================================================================
def per_frequency_correction(cwt_results, baseline_start=None, baseline_end=None,
                             task_window=None, baseline_per_freq=None):
    """
    Apply baseline correction at the frequency level (before band integration).
    For each frequency, compute the mean power across the baseline window,
    then derive diff / pct_change / log_ratio at every (frequency, time) point
    referenced against that frequency's own baseline mean.

    baseline_per_freq, if given, is used directly as the per-frequency
    reference — e.g. precomputed from a separate baseline TRIAL's own CWT,
    whose time axis doesn't overlap this cwt_results' own 'times'.
    baseline_start/baseline_end are then ignored. Otherwise the baseline
    mean is computed by time-masking this same cwt_results matrix.

    Returns
    -------
    cwt_results with added keys:
        'power_diff'        : 2-D (n_freqs, n_times)
        'power_pct_change'  : 2-D (n_freqs, n_times)
        'power_log_ratio'   : 2-D (n_freqs, n_times)
        'baseline_per_freq' : 1-D (n_freqs,)
        'baseline_window'   : (start, end) or None (None when baseline_per_freq was supplied directly)
        'task_window'       : (start, end) or None
    """
    f     = cwt_results['frequencies']
    t     = cwt_results['times']
    power = cwt_results['power']
    n_f   = len(f)
    n_t   = len(t)

    if baseline_per_freq is None:
        bl_mask = (t >= baseline_start) & (t < baseline_end)

        if not np.any(bl_mask):
            print(f"  WARNING: no samples in baseline window "
                  f"[{baseline_start}, {baseline_end}) — skipping correction")
            return cwt_results

        # Per-frequency baseline mean (one scalar per frequency row)
        baseline_per_freq = np.nanmean(power[:, bl_mask], axis=1)
        baseline_window = (baseline_start, baseline_end)
    else:
        baseline_per_freq = np.asarray(baseline_per_freq, dtype=float)
        baseline_window = None

    if task_window is not None:
        task_mask = (t >= task_window[0]) & (t < task_window[1])
    else:
        task_mask = np.ones(n_t, dtype=bool)

    # Initialise output matrices as NaN
    power_diff = np.full((n_f, n_t), np.nan)
    power_pct  = np.full((n_f, n_t), np.nan)
    power_log  = np.full((n_f, n_t), np.nan)

    # Column vector for broadcasting across time axis
    bl_col = baseline_per_freq[:, None]

    # Frequencies with valid (positive, non-NaN) baseline
    valid_freq = ~(np.isnan(baseline_per_freq) | (baseline_per_freq <= 0))
    valid_freq_2d = valid_freq[:, None] & task_mask[None, :]

    # diff: defined wherever baseline is not NaN
    diff_defined = ~np.isnan(baseline_per_freq)
    diff_mask_2d = diff_defined[:, None] & task_mask[None, :]
    power_diff[diff_mask_2d] = (power - bl_col)[diff_mask_2d]

    # pct_change & log_ratio: require positive baseline
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_full = (power - bl_col) / bl_col * 100
        log_full = np.where(power > 0, np.log(power / bl_col), np.nan)

    power_pct[valid_freq_2d] = pct_full[valid_freq_2d]
    power_log[valid_freq_2d] = log_full[valid_freq_2d]

    cwt_results['power_diff']        = power_diff
    cwt_results['power_pct_change']  = power_pct
    cwt_results['power_log_ratio']   = power_log
    cwt_results['baseline_per_freq'] = baseline_per_freq
    cwt_results['baseline_window']   = baseline_window
    cwt_results['task_window']       = task_window

    return cwt_results


# ==========================================================================
# BAND POWER EXTRACTION
# ==========================================================================
def extract_band_power_cwt(cwt_results, bands=None):
    """
    Extract band-limited power from a CWT power matrix at every time point.

    Returns
    -------
    dict with keys:
        'times'                 : array (n_samples,)
        'band_power'            : dict[band_name] -> 1-D array
        'band_power_diff'       : dict[band_name] -> 1-D array (if corrected)
        'band_power_pct_change' : dict[band_name] -> 1-D array (if corrected)
        'band_power_log_ratio'  : dict[band_name] -> 1-D array (if corrected)
        'task_window'           : (start, end) or None (passed through)
    """
    if bands is None:
        bands = HRV_BANDS

    f     = cwt_results['frequencies']
    t     = cwt_results['times']
    power = cwt_results['power']

    # Optional per-freq-corrected matrices
    power_diff = cwt_results.get('power_diff')
    power_pct  = cwt_results.get('power_pct_change')
    power_log  = cwt_results.get('power_log_ratio')

    has_corrections = power_diff is not None

    band_power      = {}
    band_power_diff = {} if has_corrections else None
    band_power_pct  = {} if has_corrections else None
    band_power_log  = {} if has_corrections else None

    for band_name, (f_low, f_high) in bands.items():
        mask = (f >= f_low) & (f < f_high)

        if not np.any(mask):
            band_power[band_name] = np.full(power.shape[1], np.nan)
            if has_corrections:
                band_power_diff[band_name] = np.full(power.shape[1], np.nan)
                band_power_pct[band_name]  = np.full(power.shape[1], np.nan)
                band_power_log[band_name]  = np.full(power.shape[1], np.nan)
            continue

        band_power[band_name] = _trapz(power[mask, :], f[mask], axis=0)

        if has_corrections:
            band_power_diff[band_name] = _trapz(power_diff[mask, :], f[mask], axis=0)
            band_power_pct[band_name]  = _trapz(power_pct[mask, :],  f[mask], axis=0)
            band_power_log[band_name]  = _trapz(power_log[mask, :],  f[mask], axis=0)

    out = {
        'times':      t,
        'band_power': band_power,
        'task_window': cwt_results.get('task_window'),
    }
    if has_corrections:
        out['band_power_diff']       = band_power_diff
        out['band_power_pct_change'] = band_power_pct
        out['band_power_log_ratio']  = band_power_log

    return out


# ==========================================================================
# BAND RESULTS → LONG-FORMAT DATAFRAME
# ==========================================================================
def band_results_to_df(band_results, participant_id,
                       drop_outside_task_window=False):
    
    representations = [
        ('raw',        'band_power'),
        ('diff',       'band_power_diff'),
        ('pct_change', 'band_power_pct_change'),
        ('log_ratio',  'band_power_log_ratio'),
    ]

    rows = []
    t = band_results['times']

    for rep_label, band_key in representations:
        band_dict = band_results.get(band_key)

        if band_dict is None and rep_label != 'raw':
            continue

        if band_dict is not None:
            for band_name, vals in band_dict.items():
                for ti, v in zip(t, vals):
                    rows.append({
                        'participant_id': participant_id,
                        'time_seconds':   ti,
                        'band':           band_name,
                        'value_type':     rep_label,
                        'power':          v,
                    })

    df = pd.DataFrame(rows)

    if drop_outside_task_window and band_results.get('task_window') is not None:
        tw_start, tw_end = band_results['task_window']
        keep = (df['time_seconds'] >= tw_start) & (df['time_seconds'] < tw_end)
        df = df[keep].reset_index(drop=True)

    return df


# ==========================================================================
# PIPELINE: COMPUTE / TASK
# ==========================================================================
def run_cwt_compute(intervals_filtered, t_resampled, fs,
                    freq_min=DEFAULT_FREQ_MIN,
                    freq_max=DEFAULT_FREQ_MAX,
                    n_freqs=DEFAULT_N_FREQS,
                    n_cycles_low=DEFAULT_N_CYCLES_LOW,
                    n_cycles_high=DEFAULT_N_CYCLES_HIGH,
                    pad_seconds=DEFAULT_PAD_SECONDS,
                    verbose=True):
    """
    Per-participant CWT computation: Wavelet convolution 
    """

    if verbose:
        print(f"\n[CWT compute — full recording]")

    cwt_results = compute_cwt_power(
        intervals_filtered, t_resampled, fs,
        freq_min=freq_min, freq_max=freq_max, n_freqs=n_freqs,
        n_cycles_low=n_cycles_low, n_cycles_high=n_cycles_high,
        pad_seconds=pad_seconds,
        verbose=verbose,
    )
    return cwt_results


def run_cwt_task(cwt_results, task, participant_id,
                 task_window, baseline_window=None,
                 baseline_per_freq=None,
                 task_intervals=None,
                 bands=None,
                 drop_outside_task_window=True,
                 verbose=True):
    """
    Per-task CWT processing.
    - Takes the pre-computed CWT power matrix
    - Apply per-frequency baseline correction, either against a time-masked
      window of this same matrix (baseline_window) or a precomputed
      per-frequency reference from elsewhere (baseline_per_freq — e.g. a
      separate baseline TRIAL's own CWT, whose time axis doesn't overlap
      this one's)
    - Integrate into bands
    - Convert to a long-format DataFrame.


    Returns
    -------
    dict with keys:
        'cwt_results'  : the deepcopied + corrected cwt_results dict
        'band_results' : output of extract_band_power_cwt
        'band_df'      : long-format DataFrame (output of band_results_to_df)
    """
    if verbose:
        print(f"\n[CWT task — {task}]")
        if baseline_window is not None:
            print(f"  Baseline window: {baseline_window[0]:.1f}-{baseline_window[1]:.1f}s")
        elif baseline_per_freq is not None:
            print(f"  Baseline: precomputed per-frequency reference")
        print(f"  Task window:     {task_window[0]:.1f}-{task_window[1]:.1f}s")

    # Deepcopy so noise and arith corrections don't interfere
    cwt_task = copy.deepcopy(cwt_results)

    # Per-frequency baseline correction
    cwt_task = per_frequency_correction(
        cwt_task,
        baseline_start=baseline_window[0] if baseline_window is not None else None,
        baseline_end=baseline_window[1] if baseline_window is not None else None,
        task_window=task_window,
        baseline_per_freq=baseline_per_freq,
    )

    # Band integration (raw + corrected matrices)
    band_results = extract_band_power_cwt(cwt_task, bands=bands)

    # Convert to long-format DataFrame
    band_df = band_results_to_df(
        band_results,
        participant_id=participant_id,
        drop_outside_task_window=drop_outside_task_window,
    )

    if verbose:
        print(f"  Band DataFrame: {len(band_df)} rows, "
              f"bands={list(band_results['band_power'].keys())}")


    return {
        'cwt_results':  cwt_task,
        'band_results': band_results,
        'band_df':      band_df,
    }


# ==========================================================================
# PLOTTING
# ==========================================================================
def plot_cwt_scalogram(cwt_results, intervals_filtered, t_resampled,
                       participant_id=None, event_times=None, event_labels=None,
                       save_path=None, show=True):
    """
    Two-panel figure: filtered RRI signal on top, CWT scalogram on bottom.
    Uses the raw (uncorrected) power matrix for display.
    """
    f     = cwt_results['frequencies']
    t     = cwt_results['times']
    power = cwt_results['power']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # --- top: signal ---
    ax1.plot(t_resampled, intervals_filtered,
             linewidth=1, color="#1f77b4", alpha=0.8)
    ax1.set_ylabel("RR Interval (filtered, ms)", fontsize=11)
    title = "Filtered RRI Signal"
    if participant_id is not None:
        title += f" — {participant_id}"
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    if event_times is not None:
        for i, t_ev in enumerate(event_times):
            label = (event_labels[i]
                     if (event_labels and i < len(event_labels)) else None)
            ax1.axvline(t_ev, color="black", linestyle="--",
                        linewidth=1.2, alpha=0.7, label=label)
        if event_labels:
            ax1.legend(loc="upper right", fontsize=9)

    # --- bottom: scalogram ---
    power_plot = np.maximum(power, 1e-20)
    pos_min = (power_plot[power_plot > 0].min()
               if (power_plot > 0).any() else 1e-20)

    ax2.pcolormesh(t, f, power_plot, shading="auto", cmap="viridis",
                   norm=LogNorm(vmin=pos_min, vmax=power_plot.max()))

    if event_times is not None:
        for t_ev in event_times:
            ax2.axvline(t_ev, color="white", linestyle="--",
                        linewidth=1.2, alpha=0.8)

    ax2.set_yscale("log")
    ax2.set_ylim([f[0], f[-1]])
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Frequency (Hz)", fontsize=11)
    ax2.set_title("CWT Scalogram — Adaptive Morlet (log power)",
                  fontsize=12, fontweight="bold")

    # Band guide lines
    for f_band in (HRV_BANDS['VLF'][0], HRV_BANDS['LF'][0],
                   HRV_BANDS['HF'][0], HRV_BANDS['HF'][1]):
        ax2.axhline(f_band, color="white", linestyle=":",
                    linewidth=1, alpha=0.6)

    ax2.text(t[0] + 30,
             np.sqrt(HRV_BANDS['VLF'][0] * HRV_BANDS['VLF'][1]),
             "VLF", color="white", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="black", alpha=0.4))
    ax2.text(t[0] + 30,
             np.sqrt(HRV_BANDS['LF'][0] * HRV_BANDS['LF'][1]),
             "LF", color="white", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="black", alpha=0.4))
    ax2.text(t[0] + 30,
             np.sqrt(HRV_BANDS['HF'][0] * HRV_BANDS['HF'][1]),
             "HF", color="white", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="black", alpha=0.4))

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
        print(f"  Saved scalogram: {save_path}.pdf")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_cwt_band_power(band_results, participant_id=None,
                        event_times=None, event_labels=None,
                        save_path=None, show=True):
    """
    Multi-panel figure: one panel per HRV band (VLF, LF, HF) showing
    raw band power evolution over time.
    """
    t  = band_results['times']
    bp = band_results['band_power']

    band_colors = {'VLF': '#ff9999', 'LF': '#99ccff', 'HF': '#99ff99'}
    bands_to_plot = [b for b in ('VLF', 'LF', 'HF') if b in bp]

    n_panels = len(bands_to_plot)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 2.5 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, band_name in zip(axes, bands_to_plot):
        color = band_colors.get(band_name, 'tab:blue')
        ax.plot(t, bp[band_name], linewidth=2, color=color, alpha=0.9)
        ax.fill_between(t, 0, bp[band_name], alpha=0.3, color=color)
        ax.set_ylabel(f'{band_name} Power', fontsize=11)
        ax.grid(True, alpha=0.3)

        if event_times is not None:
            for t_ev in event_times:
                ax.axvline(t_ev, color='black', linestyle='--',
                           linewidth=1, alpha=0.5)

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    # Event labels on last axis
    if event_times is not None and event_labels is not None:
        for i, t_ev in enumerate(event_times):
            label = event_labels[i] if i < len(event_labels) else None
            axes[-1].axvline(t_ev, color='black', linestyle='--',
                             linewidth=1, alpha=0.5, label=label)
        axes[-1].legend(fontsize=9, loc='upper right')

    suptitle = 'CWT — Band Power Evolution'
    if participant_id is not None:
        suptitle += f' ({participant_id})'
    plt.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
        print(f"  Saved band power: {save_path}.pdf")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig