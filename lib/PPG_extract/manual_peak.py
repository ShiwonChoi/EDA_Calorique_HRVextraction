import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from lib.config import *


def load_corrected_rri_for_participant(participant_path, participant_id,
                                       use_physio=False, use_stat=False):
    """
    Find and load the one corrected-RRI CSV for this participant's whole
    continuous recording.

    Unlike the old per-trial-file layout (one shimmer file + one corrected
    CSV per trial), this format has exactly one shimmer file per participant,
    so the corrected CSV — if one exists — is named after that same shimmer
    file (e.g. Processed_PPG/SC_pilot/<shimmer_stem>..._rr_intervals_corrigé.csv),
    not per trial/condition.

    Returns:
        df_corr (pd.DataFrame | None), base_rri (int) — same shape as
        load_corrected_rri; (None, 0) if no shimmer or corrected file is found.
    """
    participant_path = Path(participant_path)
    corr_dir = CORR_DIR / participant_path.name
    if not corr_dir.exists():
        print(f"  No corrected dir at {corr_dir} — using auto peaks only")
        return None, 0

    shimmer_files = sorted(participant_path.glob("shimmer_*.csv"))
    if len(shimmer_files) != 1:
        print(f"  Expected exactly one shimmer_*.csv in {participant_path}, "
              f"found {len(shimmer_files)} — using auto peaks only")
        return None, 0

    shimmer_stem = shimmer_files[0].stem.lower()
    candidates = [
        p for p in corr_dir.rglob("*.csv")
        if p.stem.lower().startswith(shimmer_stem) and "corrig" in p.stem.lower()
    ]
    if not candidates:
        print(f"  No corrected CSV matching '{shimmer_stem}*corrig*' in {corr_dir} "
              f"— using auto peaks only")
        return None, 0
    if len(candidates) > 1:
        print(f"  WARNING: multiple corrected CSVs matched — using {candidates[0].name}")
    else:
        print(f"  Corrected CSV: {candidates[0].name}")

    return load_corrected_rri(candidates[0], use_physio=use_physio, use_stat=use_stat)


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
        peak1_index,  peak2_index      (sample index in raw recording)
        heart_rate_bpm,
        peak1_classification, peak2_classification  ('AUTO' | 'MANUAL' | 'BAD')
        physiologically_valid          (True / False)
        statistically_valid            (True / False)

    Classification filter (always applied):
        An interval is kept only when BOTH peak1 and peak2 classification != 'BAD'.

    Args:
        csv_path:   Direct path to the corrected RRI CSV.
        use_physio: Also keep only physiologically_valid == True rows.
        use_stat:   Also keep only statistically_valid   == True rows.

    Returns:
        Filtered DataFrame, or None on read error.
    """
    try:
        df = pd.read_csv(Path(csv_path))
    except Exception as e:
        print(f"    Error reading corrected RRI file: {e}")
        return None, 0

    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}

    # Total intervals in the file before any filtering
    base_rri = len(df)

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

    return df, base_rri


def detect_peaks_full(df_ppg, fs):
    """
    Run NK2 PPG peak detection ONCE on the whole continuous session (not
    per trial — a short trial slice, e.g. a brief baseline window, can have
    too few samples/peaks for NK2's signal-quality step and raise).

    Returns:
        signal (pd.DataFrame) : NK2 signal with PPG_Peaks, plus PPG_Peaks_Corr
                                 initialised to 0 (filled in by place_corrected_peaks).
        info   (dict)         : NK2 info dict.
    """
    participant_id = df_ppg["participant"].iloc[0] if "participant" in df_ppg.columns else "unknown"
    print(f"\nDetecting PPG peaks for {participant_id} — full recording ({len(df_ppg)} samples)...")

    signal, info = nk.ppg_process(df_ppg["PPG"], fs)
    signal["PPG_Peaks_Corr"] = 0

    n_auto = int(signal["PPG_Peaks"].sum())
    print(f"  {n_auto} auto peaks detected across full recording")

    return signal, info


def place_corrected_peaks(signal, df_ppg, df_corr, use_physio=False, use_stat=False):
    """
    Mark PPG_Peaks_Corr=1 on `signal` at the sample nearest each corrected
    peak time in df_corr. df_corr (from load_corrected_rri_for_participant)
    already covers the whole continuous recording, so no per-trial epoch
    restriction is needed here — unlike the old per-trial version, there's
    only one corrected file to place, once.
    """
    if df_corr is None or len(df_corr) == 0:
        return signal

    col_map = {c.lower(): c for c in df_corr.columns}
    t1_col  = _find_col(col_map, "peak1_time_s")
    t2_col  = _find_col(col_map, "peak2_time_s")
    if not (t1_col and t2_col):
        print("  WARNING: corrected CSV missing peak1_time_s/peak2_time_s — skipping placement")
        return signal

    rel_ms = df_ppg["rel_zero_ref (ms)"].values
    all_times_s = np.unique(np.concatenate([
        df_corr[t1_col].dropna().values,
        df_corr[t2_col].dropna().values,
    ]))
    corr_rel_ms = all_times_s * 1000
    in_range    = (corr_rel_ms >= rel_ms[0]) & (corr_rel_ms <= rel_ms[-1])
    print(f"  Corrected peaks placed: {in_range.sum()}/{len(corr_rel_ms)}")

    corr_col = signal.columns.get_loc("PPG_Peaks_Corr")
    for rel in corr_rel_ms[in_range]:
        nearest = np.argmin(np.abs(rel_ms - rel))
        signal.iat[nearest, corr_col] = 1

    return signal


def extract_beats(signal, df_ppg):
    """
    Beat times (s) and RR intervals (ms) for the whole continuous recording.

    Uses corrected peaks (PPG_Peaks_Corr) when any were placed, else falls
    back to NK2's auto-detected peaks (PPG_Peaks) — participants without a
    corrected CSV (e.g. SC_pilot today) still get RRI computed from auto peaks
    instead of an all-zero PPG_Peaks_Corr column.

    Returns:
        beat_times (np.ndarray) : time (s) of each beat except the last.
        rri_values (np.ndarray) : RR interval (ms) between successive beats.
    """
    if int(signal["PPG_Peaks_Corr"].sum()) > 0:
        peak_idx = np.where(signal["PPG_Peaks_Corr"].values == 1)[0]
        print(f"  Using {len(peak_idx)} corrected peaks for beat extraction")
    else:
        peak_idx = np.where(signal["PPG_Peaks"].values == 1)[0]
        print(f"  Using {len(peak_idx)} auto peaks for beat extraction (no corrected peaks)")

    peak_times = df_ppg["time_seconds"].values[peak_idx]
    if len(peak_times) > 1:
        rri_values = np.diff(peak_times) * 1000
        beat_times = peak_times[:-1]
    else:
        rri_values = np.array([])
        beat_times = np.array([])

    return beat_times, rri_values


def plot_corr_peaks(signal, df_ppg, df_events, df_corr, participant, show=True):
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
