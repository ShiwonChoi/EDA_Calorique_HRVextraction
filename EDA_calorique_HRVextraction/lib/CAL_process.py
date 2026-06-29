import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg
from lib.PPG_extract.manual_peak import load_corrected_peaks, process_rri_intervals, plot_corr_peaks
from lib.Metric_extraction.RRI_preprocess import preprocess_pipeline, preprocess_visualize, plot_preprocessing_steps, validate_intervals

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


            # ── 2. NK2 peak detection + event-window RRI ───────────────────
            corr_paths_t = load_corrected_peaks(participant_path, participant_id, df_ppg_t)
            rri_ms_t, rri_trial_t, corr_t, signal_t, info_t, epoch_t = process_rri_intervals(
                df_ppg_t, df_events_t, fs,
                corr_paths=corr_paths_t,
                use_physio=use_physio,
                use_stat=use_stat,
            )


            # ── 3. Concatenate signal_t with time columns and metadata ──────────
            # Attach the PPG time axis so signal_t is self-contained
            signal_t["time_seconds"] = df_ppg_t["time_seconds"].values
            signal_t["rel_zero_ref (ms)"] = df_ppg_t["rel_zero_ref (ms)"].values
            signal_t["abs_zero_ref (ms)"] = df_ppg_t["abs_zero_ref (ms)"].values
            # Metadata labels
            signal_t["participant"] = participant_id
            signal_t["trial"]       = trial
            signal_t["condition"]   = df_ppg_t["condition"].iloc[0]

            
            # ── 4. Visualise corrected vs auto peaks ───────────────────────
            if show:
                plot_corr_peaks(signal_t, df_ppg_t, df_events_t, corr_t, participant_id, show=show)

            signal_list.append(signal_t)
            print(f"  {trial} done — signal shape: {signal_t.shape}")


            # ── 5. Preprocess RRI ───────────────────────
            peak_indices = np.where(signal_t["PPG_Peaks_Corr"] == 1)[0]
            peak_times = signal_t["time_seconds"].iloc[peak_indices].values
            rri_values = np.diff(peak_times) * 1000 # Convert seconds to ms, length = n_peaks-1
            beat_times = peak_times[:-1]

            # Create RRI column for signal_clean
            rri_col = np.full(len(signal_t), np.nan)
            for i in range(len(rri_values)):
                start_idx = peak_indices[i]
                end_idx = peak_indices[i + 1]
                rri_col[start_idx:end_idx] = rri_values[i]

            signal_t["RRI"] = rri_col
            print(f"RRI col created: {np.sum(~np.isnan(rri_col))} valid samples")


            # Resampling : Artifact detection, removal and interpolation
            # -------------------------------------------------------------------------------------------
            results = preprocess_visualize(beat_times, rri_values, use_physio=use_physio, use_stat=use_stat)

            # Visualize each preprocessing step
            print("\nCreating tachogram from raw RRI to filtered RRI")
            plot_preprocessing_steps(results, participant_id=participant_id,
                                      df_events=df_events_t, trial=trial, show=True)

            print("\nPreprocessing complete.")
            print(f"  Input:  {len(results['intervals_raw'])} beats (irregular)")
            print(f"  Output: {len(results['intervals_clean'])} beats cleaned")
            print(f"  Output: {len(results['intervals_resampled'])} resampled beats")
            print(f"  Output: {len(results['intervals_filtered'])} samples at {results['fs_resample']} Hz (uniform, filtered)")

            # Quality Validation
            # -------------------------------------------------------------------------------------------
            print("\n" + "=" * 60)
            print("QUALITY VALIDATION REPORT")
            print("we expect highpass filter to change inter-beat intervals")
            print(" to filtered residuals that are now centered to 0")
            print("=" * 60)

            stages = {
                "1. Raw intervals":           validate_intervals(results['intervals_raw']),
                "2. After artifact removal":  validate_intervals(results['intervals_clean']),
                "3. After resampling (4 Hz)": validate_intervals(results['intervals_resampled']),
                "4. After high-pass filter":  validate_intervals(results['intervals_filtered']),
            }

            for stage_name, stats in stages.items():
                print(f"\n{stage_name}:")
                print("-" * 40)
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            

            # ── 6. Extract global HRV metrics ───────────────────────
            # TODO : Extact HRV metrics


            # ── 7. Extract Interval HRV metrics ───────────────────────
            # TODO : Extract HRV metrics 


        # TODO : combine the abs zero ref timeline between trials to create a continuous data

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
