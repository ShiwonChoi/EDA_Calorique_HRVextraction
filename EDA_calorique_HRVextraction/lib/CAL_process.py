import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg
from lib.PPG_extract.manual_peak import load_corrected_peaks, process_rri_intervals, plot_corr_peaks


def full_process_single(participant_path, use_physio=True, use_stat=False, show=False):
    """
    Process one participant's PPG data on a per-trial basis.

    For each trial (Trial00, Trial01, Trial02) the pipeline runs independently:
        load_and_clean_ppg  → load_corrected_peaks → process_rri_intervals → plot_corr_peaks

    Per-trial results are collected in signal_list.  Each entry is the NK2 signal
    DataFrame enriched with time columns and participant / trial / condition labels so
    entries are self-contained for downstream analysis.  Concatenation into a single
    combined DataFrame is addressed separately once per-trial processing is complete.

    Returns:
        signal_list  (list[pd.DataFrame]) : One DataFrame per trial.
        participant_id (str)              : Participant identifier (e.g. "P002").
    """
    participant_path = Path(participant_path)

    # Discover available trials by listing shimmer files
    stress_dir = participant_path / "Stress measures"
    # stem = "shimmer_P002_Trial00_baseline_YYYYMMDD_HHMMSS" → split index 2 = trial
    trials = sorted({
        f.stem.split("_")[2]
        for f in stress_dir.glob("shimmer_*.csv")
    })
    print(f"\nParticipant folder : {participant_path.name}")
    print(f"Trials found       : {trials}")

    signal_list   = []
    participant_id = None

    try:
        for trial in trials:
            print(f"\n{'─' * 55}")
            print(f"  Trial: {trial}")
            print(f"{'─' * 55}")

            # ── 1. Load and preprocess (this trial only) ──────────────────
            df_ppg_t, df_events_t, fs, _, participant_id = load_and_clean_ppg(
                participant_path, trial_filter=trial, show=show
            )

            # ── 2. Find corrected RRI CSV for this trial ───────────────────
            corr_paths_t = load_corrected_peaks(participant_path, participant_id, df_ppg_t)

            # ── 3. NK2 peak detection + event-window RRI ───────────────────
            rri_ms_t, rri_trial_t, corr_t, signal_t, info_t, epoch_t = process_rri_intervals(
                df_ppg_t, df_events_t, fs,
                corr_paths=corr_paths_t,
                use_physio=use_physio,
                use_stat=use_stat,
            )

            # ── 4. Enrich signal_t with time columns and metadata ──────────
            # Attach the PPG time axis so signal_t is self-contained
            signal_t["time_seconds"] = df_ppg_t["time_seconds"].values
            signal_t["rel_zero_ref"] = df_ppg_t["rel_zero_ref"].values
            signal_t["abs_zero_ref"] = df_ppg_t["abs_zero_ref"].values
            # Metadata labels
            signal_t["participant"] = participant_id
            signal_t["trial"]       = trial
            signal_t["condition"]   = df_ppg_t["condition"].iloc[0]

            # ── 5. Visualise corrected vs auto peaks ───────────────────────
            if show:
                plot_corr_peaks(signal_t, df_ppg_t, df_events_t, corr_t, participant_id, show=show)

            signal_list.append(signal_t)
            print(f"  {trial} done — signal shape: {signal_t.shape}")

        print(f"\n{'=' * 55}")
        print(f"  All trials processed for {participant_id}.")
        print(f"  signal_list length: {len(signal_list)}")
        print(f"{'=' * 55}")

        return signal_list, participant_id

    except Exception as e:
        import traceback
        print(f"\nERROR processing {participant_path.name}: {e}")
        traceback.print_exc()
        return [], participant_id
