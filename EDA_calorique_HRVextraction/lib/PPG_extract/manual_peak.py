import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from lib.config import *


def load_corrected_peaks(participant_path, participant_id, df_ppg):
    """
    Find corrected RRI CSVs for every trial under CORR_DIR / <participant_folder>.

    Returns:
        {trial: Path} for each trial whose corrected CSV was found, or {} if the
        participant folder does not exist in CORR_DIR.
    """
    corr_dir = CORR_DIR / Path(participant_path).name
    if not corr_dir.exists():
        print(f"  No corrected dir at {corr_dir} — using auto peaks only")
        return {}

    trial_condition = (
        df_ppg[["trial", "condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    corr_paths = {}
    for trial, condition in sorted(trial_condition):
        candidates = list(corr_dir.rglob(f"*{participant_id}*{trial}*{condition}*.csv"))
        if candidates:
            if len(candidates) > 1:
                print(f"  WARNING: multiple corrected CSVs for {trial} — using {candidates[0].name}")
            else:
                print(f"  Corrected CSV ({trial}): {candidates[0].name}")
            corr_paths[trial] = candidates[0]
        else:
            print(f"  No corrected CSV for {trial} ({condition})")

    return corr_paths


def _find_col(col_map, *candidates):
    """Return the first column name (original case) matching any candidate (case-insensitive)."""
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    return None


def _as_bool(series):
    """Coerce a column that may be native bool or string 'True'/'False' to bool."""
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() == "true")


def load_corrected_rri(csv_path, use_physio=False, use_stat=False):
    """
    Load and filter a corrected RRI CSV given its direct path.

    Expected CSV columns:
        interval_index, rr_interval_ms,
        peak1_time_s, peak2_time_s     (seconds since Shimmer connected)
        peak1_index,  peak2_index      (sample index in raw trial recording)
        heart_rate_bpm,
        peak1_classification, peak2_classification  ('AUTO' | 'MANUAL' | 'BAD')
        physiologically_valid          (True / False)
        statistically_valid            (True / False)

    Classification filter (always applied):
        An interval is kept only when BOTH peak1 and peak2 classification != 'BAD'.

    Args:
        csv_path:   Direct path to the corrected RRI CSV (resolved by load_corrected_peaks).
        use_physio: Also keep only physiologically_valid == True rows.
        use_stat:   Also keep only statistically_valid   == True rows.

    Returns:
        Filtered DataFrame, or None on read error.
    """
    try:
        df = pd.read_csv(Path(csv_path))
    except Exception as e:
        print(f"    Error reading corrected RRI file: {e}")
        return None

    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}

    # ── Classification filter (always applied) ────────────────────────────────
    class1_col = _find_col(col_map, "peak1_classification")
    class2_col = _find_col(col_map, "peak2_classification")
    single_col = _find_col(col_map, "classification")

    if class1_col and class2_col:
        n_before = len(df)
        mask = (
            (df[class1_col].str.strip().str.upper() != "BAD") &
            (df[class2_col].str.strip().str.upper() != "BAD")
        )
        df = df[mask].reset_index(drop=True)
        print(f"    Classification filter: {n_before - len(df)} bad intervals removed, {len(df)} kept")
    elif single_col:
        n_before = len(df)
        df = df[df[single_col].str.strip().str.upper() != "BAD"].reset_index(drop=True)
        print(f"    Classification filter: {n_before - len(df)} bad intervals removed, {len(df)} kept")
    else:
        print(f"    WARNING: No classification column found — classification filter skipped")

    # ── Physiologically valid filter (optional) ───────────────────────────────
    if use_physio:
        physio_col = _find_col(col_map, "physiologically_valid", "physio_valid")
        if physio_col:
            n_before = len(df)
            df = df[_as_bool(df[physio_col])].reset_index(drop=True)
            print(f"    Physio filter: {n_before - len(df)} intervals removed, {len(df)} kept")
        else:
            print(f"    WARNING: use_physio=True but no physiologically_valid column found")

    # ── Statistically valid filter (optional) ─────────────────────────────────
    if use_stat:
        stat_col = _find_col(col_map, "statistically_valid", "stat_valid")
        if stat_col:
            n_before = len(df)
            df = df[_as_bool(df[stat_col])].reset_index(drop=True)
            print(f"    Stat filter: {n_before - len(df)} intervals removed, {len(df)} kept")
        else:
            print(f"    WARNING: use_stat=True but no statistically_valid column found")

    return df


def process_rri_intervals(df_ppg, df_events, fs, corr_paths=None,
                          use_physio=False, use_stat=False):
    """
    Process RRI intervals for a single trial.

    Returns:
        rri_auto_ms    (np.ndarray)   : Within-event-window auto RRI in ms.
        rri_auto_trial (np.ndarray)   : Trial label per RRI interval.
        corr_by_trial  (dict)         : {trial: event-window-filtered corrected RRI DataFrame}.
        signal         (pd.DataFrame) : NK2 signal with PPG_Peaks and PPG_Peaks_Corr.
        info           (dict)         : NK2 info dict.
        epoch_bounds   (dict)         : {trial: {"rel_lo","rel_hi","abs_lo","abs_hi"}} in ms.
    """
    participant_id = df_ppg["participant"].iloc[0] if "participant" in df_ppg.columns else "unknown"
    trial     = df_ppg["trial"].iloc[0]
    trial_rel = df_ppg["rel_zero_ref (ms)"].values
    print(f"\nProcessing PPG signal for {participant_id} — {trial}...")

    # Event window bounds
    epoch_bounds = {}
    ev_ms = df_events.loc[df_events["trial"] == trial, "time_since_connected_ms"]
    if not ev_ms.empty:
        t_start = float(df_ppg["rel_zero_ref (ms)"].iloc[0])
        first_ev, last_ev = float(ev_ms.iloc[0]), float(ev_ms.iloc[-1])
        epoch_bounds[trial] = {
            "rel_lo": first_ev, "rel_hi": last_ev,
            "abs_lo": t_start + first_ev, "abs_hi": t_start + last_ev,
        }
        print(f"  Event window: [{first_ev/1000:.3f}s – {last_ev/1000:.3f}s]")

    if not corr_paths:
        print("  No corrected paths provided — using auto peaks only")

    # NK2 peak detection
    signal, info = nk.ppg_process(df_ppg["PPG"], fs)
    signal["PPG_Peaks_Corr"] = 0
    auto_idx = np.where(signal["PPG_Peaks"].values == 1)[0]

    # Load corrected RRI, filter to event window, place peaks
    corr_by_trial = {}
    if corr_paths and trial in corr_paths:
        print(f"  Loading corrected RRI — {trial}...")
        df_corr = load_corrected_rri(corr_paths[trial], use_physio=use_physio, use_stat=use_stat)
        if df_corr is not None:
            col_map = {c.lower(): c for c in df_corr.columns}
            t1_col  = _find_col(col_map, "peak1_time_s")
            t2_col  = _find_col(col_map, "peak2_time_s")
            if t1_col and t2_col:
                if trial in epoch_bounds:
                    lo, hi  = epoch_bounds[trial]["rel_lo"], epoch_bounds[trial]["rel_hi"]
                    in_ep   = (
                        (df_corr[t1_col] * 1000 >= lo) & (df_corr[t1_col] * 1000 <= hi) &
                        (df_corr[t2_col] * 1000 >= lo) & (df_corr[t2_col] * 1000 <= hi)
                    )
                    n_before = len(df_corr)
                    df_corr  = df_corr[in_ep].reset_index(drop=True)
                    removed  = n_before - len(df_corr)
                    if removed:
                        print(f"    Epoch filter: {removed} intervals outside event window removed")

                all_times_s = np.unique(np.concatenate([
                    df_corr[t1_col].dropna().values,
                    df_corr[t2_col].dropna().values,
                ]))
                corr_rel_ms = all_times_s * 1000
                in_range    = (corr_rel_ms >= trial_rel[0]) & (corr_rel_ms <= trial_rel[-1])
                print(f"    Corrected peaks placed: {in_range.sum()}/{len(corr_rel_ms)}")
                for rel_ms in corr_rel_ms[in_range]:
                    nearest = np.argmin(np.abs(trial_rel - rel_ms))
                    signal.iat[nearest, signal.columns.get_loc("PPG_Peaks_Corr")] = 1

            corr_by_trial[trial] = df_corr

    # Event-window filter for auto peaks and RRI
    if trial in epoch_bounds:
        lo, hi     = epoch_bounds[trial]["rel_lo"], epoch_bounds[trial]["rel_hi"]
        in_window  = (trial_rel[auto_idx] >= lo) & (trial_rel[auto_idx] <= hi)
        auto_idx_w = auto_idx[in_window]
    else:
        auto_idx_w = auto_idx

    peak_times_w   = df_ppg["time_seconds"].values[auto_idx_w]
    rri_auto_ms    = np.diff(peak_times_w) * 1000 if len(peak_times_w) > 1 else np.array([])
    rri_auto_trial = np.full(len(rri_auto_ms), trial)

    print(f"  {len(auto_idx)} auto peaks, {len(auto_idx_w)} in event window, "
          f"{len(rri_auto_ms)} RRI intervals")
    print(f"  Corrected peaks in signal: {int(signal['PPG_Peaks_Corr'].sum())}")

    return rri_auto_ms, rri_auto_trial, corr_by_trial, signal, info, epoch_bounds


def plot_corr_peaks(signal, df_ppg, df_events, corr_by_trial, participant, show=True):
    time     = df_ppg["time_seconds"].values
    ppg      = signal["PPG_Clean"].values
    auto_idx = np.where(signal["PPG_Peaks"].values == 1)[0]
    corr_idx = np.where(signal["PPG_Peaks_Corr"].values == 1)[0]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    axes[0].plot(time, ppg, color="black", linewidth=1)
    axes[0].scatter(time[auto_idx], ppg[auto_idx], color="green", s=30, label="Auto")
    axes[0].set_title(f"{participant} — Auto Peaks")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    axes[1].plot(time, ppg, color="black", linewidth=1)
    axes[1].scatter(time[corr_idx], ppg[corr_idx], color="blue", s=30, label="Corrected")
    axes[1].set_title(f"{participant} — Corrected Peaks")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    plt.tight_layout()
    if show:
        plt.show()

    return fig
