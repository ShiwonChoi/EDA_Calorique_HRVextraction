import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg
from lib.PPG_extract.manual_peak import load_corrected_peaks, process_rri_intervals, plot_corr_peaks
from lib.Metric_extraction.RRI_preprocess import preprocess_pipeline, preprocess_visualize, plot_preprocessing_steps, validate_intervals
from lib.Metric_extraction.HRV_temp_extract import get_temp_metrics
from lib.Metric_extraction.HRV_temp_bin import bin_temp_30s
from lib.Metric_extraction.HRV_freq_extract import (
    run_cwt_compute, run_cwt_task, extract_band_power_cwt, band_results_to_df, plot_cwt_scalogram,
)
from lib.Metric_extraction.HRV_freq_bin import bin_totalbandpower, bin_bandpower_30s
from lib.Metric_extraction.HRV_df import build_result_row
from lib.config import OUTPUT_COLUMNS

def full_process_single(participant_path, use_physio=True, use_stat=False, show=False, bin=30):
    """
    Process one participant's PPG data on a per-trial basis.

    The whole session (baseline + 6 stimulus blocks) is one continuous
    recording, loaded once via load_and_clean_ppg. Trials are then the 7
    windows derived by assign_trial_condition:
        0        'baseline'
        1..6     '<sound>_<design>' e.g. 'loud_individu', 'quiet_quatre_sons'
    Block 6 has no rest_start/rest_end of its own, so its window (and
    'recovery' phase) extends through post_recovery_start/post_recovery_end
    instead — see phase_windows in HRV_temp_extract.py.
    For each trial the pipeline runs independently on its slice of the
    combined frames:
        load_corrected_peaks → process_rri_intervals → plot_corr_peaks
        → RRI preprocessing → global HRV metric extraction

    Both CAL_temp and CAL_freq use the unified OUTPUT_COLUMNS schema (config.py):
        participant, trial, condition,
        time_interval_relative, time_interval_absolute (seconds),
        task_moment, recording_type ('total' or 'interval'),
        Metric, Value_type, Value,
        sample_size ("<n_clean> of <n_raw>"), status, error

    CAL_temp collects temporal HRV rows (mean_HR, mean_RRI, RMSSD, SDNN):
        - 'total' rows: whole-trial metric, baseline-referenced against trial 0.
        - 'interval' rows (block trials only): 30-s bins, same baseline reference.

    CAL_freq collects frequency HRV rows (VLF, LF, HF band power):
        - 'total' rows: whole-trial mean per band, all trials.
        - 'interval' rows (block trials only): 30-s bin means.
        Baseline rows carry diff/pct_change/log_ratio = 0.0 by convention;
        other rows are corrected per-frequency against trial 0.

    Returns:
        participant_id (str)         : Participant identifier (e.g. "SC_01").
        df_temp        (pd.DataFrame): Temporal HRV rows, schema = OUTPUT_COLUMNS.
        df_freq        (pd.DataFrame): Frequency HRV rows, schema = OUTPUT_COLUMNS.
    """
    participant_path = Path(participant_path)
    print(f"\nParticipant folder : {participant_path.name}")

    CAL_temp              = []
    CAL_freq              = []
    baseline_temp_raw     = None
    baseline_per_freq_raw = None
    participant_id        = None

    try:
        # Load the whole continuous session once; slice per trial below.
        df_ppg, df_events, fs, _, participant_id = load_and_clean_ppg(participant_path, show=show)
        trials = sorted(df_events['trial'].dropna().unique())
        print(f"Trials found       : {trials}")

        for trial in trials:
            print(f"\n{'─' * 55}")
            print(f"  Trial: {trial}")
            print(f"{'─' * 55}")


            # ── 1. Slice this trial's rows out of the combined session ─────
            df_ppg_t    = df_ppg[df_ppg['trial'] == trial].reset_index(drop=True)
            df_events_t = df_events[df_events['trial'] == trial].reset_index(drop=True)


            # ── 2. NK2 peak detection + event-window RRI ───────────────────
            corr_paths_t = load_corrected_peaks(participant_path, participant_id, df_ppg_t)
            rri_ms_t, rri_trial_t, corr_t, signal_t, info_t, epoch_t, base_rri = process_rri_intervals(
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


            # ── 4. Visualize corrected vs auto peaks ───────────────────────
            if show:
                plot_corr_peaks(signal_t, df_ppg_t, df_events_t, corr_t, participant_id, show=show)
            print(f"  {trial} done — signal shape: {signal_t.shape}")


            # ── 5. Preprocess RRI ───────────────────────
            peak_indices = np.where(signal_t["PPG_Peaks_Corr"] == 1)[0]
            peak_times = signal_t["time_seconds"].iloc[peak_indices].values
            rri_values = np.diff(peak_times) * 1000 # Convert seconds to ms
            beat_times = peak_times[:-1]
            print(f"    RRI col created: {np.sum(~np.isnan(rri_values))} valid samples")


            # Resampling : Artifact detection, removal and interpolation
            # -------------------------------------------------------------------------------------------
            results = preprocess_visualize(beat_times, rri_values, use_physio=use_physio, use_stat=use_stat)

            # sample_size: clean / after-artifact-removal of corrected-CSV-total-before-filtering
            sample_size = f"{len(results['intervals_clean'])} / {len(results['intervals_raw'])}" #of {base_rri}"

            # Visualize each preprocessing step
            print("\nCreating tachogram from raw RRI to filtered RRI")
            plot_preprocessing_steps(results, participant_id=participant_id,
                                      df_events=df_events_t, trial=trial, show=show)

            print("\nPreprocessing complete.")
            print(f"  Input:  {len(results['intervals_raw'])} beats (irregular)")
            print(f"  Output: {len(results['intervals_clean'])} beats cleaned") # Temporal HRV
            print(f"  Output: {len(results['intervals_resampled'])} resampled beats") # Cubic spline interpolation
            print(f"  Output: {len(results['intervals_filtered'])} samples at {results['fs_resample']} Hz (uniform, filtered)") # Frequency HRV

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


            # -- 6. Global & Interval HRV metrics ---------------------------------------------------------------------------------
            condition = df_ppg_t['condition'].iloc[0]

            # Task window: first → last event for this trial (seconds since device connected)
            ev_times_s  = df_events_t['time_since_connected_ms'].values / 1000
            task_window = (float(ev_times_s[0]), float(ev_times_s[-1]))

            # task_moment for whole-trial (total) rows
            total_task_moment = 'baseline' if condition == 'baseline' else 'total'

            # -- 6a. Temporal -------------------------------------------------------------------------------------------
            metrics_temp = get_temp_metrics(results['intervals_clean'])
            if condition == 'baseline':
                baseline_temp_raw = metrics_temp

            for metric_name, metric_value in metrics_temp.items():
                bl = baseline_temp_raw[metric_name]
                CAL_temp.extend(build_result_row(
                    participant_id=participant_id,
                    trial=trial,
                    condition=condition,
                    time_interval_rel_start=0.0,
                    time_interval_abs_start=task_window[0],
                    time_interval_rel_end=task_window[1] - task_window[0],
                    time_interval_abs_end=task_window[1],
                    task_moment=total_task_moment, recording_type='total',
                    metric_name=metric_name, metric_value=metric_value,
                    baseline_mean=bl, sample_size=sample_size,
                ))
            print(f"\n  Temporal: {metrics_temp}")

            # 30-s binned temporal metrics — block trials only (baseline kept whole)
            if condition != 'baseline':
                temp_binned = bin_temp_30s(
                    results['intervals_clean'], results['beat_times_clean'],
                    trial, condition, task_window, df_events_t, participant_id,
                    baseline_temp_raw, sample_size, bin_width=bin,
                )
                CAL_temp.extend(temp_binned.to_dict('records'))

            # ── 6b. Frequency (CWT band power) ────────────────
            print("\n  Computing CWT...")
            cwt_r = run_cwt_compute(results['intervals_filtered'], results['t_resampled'],
                                    results['fs_resample'], verbose=False)

            if condition == 'baseline':
                # Trial 0 IS the baseline — raw band power only, no
                # per-frequency correction to run against itself. Its own
                # per-frequency mean (over its whole window) becomes the
                # reference other trials are corrected against below.
                band_results = extract_band_power_cwt(cwt_r)
                band_df = band_results_to_df(
                    band_results, participant_id=participant_id,
                    drop_outside_task_window=False,
                )

                bl_mask = (cwt_r['times'] >= task_window[0]) & (cwt_r['times'] < task_window[1])
                baseline_per_freq_raw = np.nanmean(cwt_r['power'][:, bl_mask], axis=1)
            else:
                # Per-frequency correction against trial 0's baseline_per_freq_raw,
                # not a time-masked window of this trial's own CWT matrix
                # (trial 0 and this trial have independent CWT time axes).
                task_out = run_cwt_task(
                    cwt_r, task=trial, participant_id=participant_id,
                    task_window=task_window,
                    baseline_per_freq=baseline_per_freq_raw,
                    verbose=False,
                )
                band_df = task_out['band_df']

            # Whole-trial mean per (band, value_type)
            total = bin_totalbandpower(
                bandpower=band_df, trial=trial, condition=condition,
                task_interval=task_window, participant_id=participant_id,
                sample_size=sample_size,
            )
            CAL_freq.extend(total.to_dict('records'))

            # 30-s binned bandpower — stim trials only (baseline kept whole)
            if condition != 'baseline':
                binned = bin_bandpower_30s(
                    bandpower=band_df, trial=trial, condition=condition,
                    task_interval=task_window, df_events_t=df_events_t,
                    participant_id=participant_id, sample_size=sample_size,
                    bin_width=bin,
                )
                CAL_freq.extend(binned.to_dict('records'))

            if show:
                ev_labels = df_events_t['event_type'].values.tolist()
                plot_cwt_scalogram(
                    cwt_r, results['intervals_filtered'], results['t_resampled'],
                    participant_id=participant_id,
                    event_times=ev_times_s.tolist(),
                    event_labels=ev_labels,
                    show=show,
                )

            print("  Freq total: " + "  ".join(
                f"{r['Metric']}.{r['Value_type']}={r['Value']:.4f}"
                for r in total.to_dict('records')
            ))

        # Output dataframes for HRV temp and freq metrics
        df_temp = pd.DataFrame(CAL_temp, columns=OUTPUT_COLUMNS) if CAL_temp else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_freq = pd.DataFrame(CAL_freq, columns=OUTPUT_COLUMNS) if CAL_freq else pd.DataFrame(columns=OUTPUT_COLUMNS)

        print(f"\n{'=' * 55}")
        print(f"  All trials processed for {participant_id}.")
        print(f"  CAL_temp rows : {len(df_temp)}")
        print(f"  CAL_freq rows : {len(df_freq)}")
        print(f"{'=' * 55}")

        return participant_id, df_temp, df_freq

    except Exception as e:
        import traceback
        print(f"\nERROR processing {participant_path.name}: {e}")
        traceback.print_exc()
        df_temp = pd.DataFrame(CAL_temp, columns=OUTPUT_COLUMNS) if CAL_temp else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_freq = pd.DataFrame(CAL_freq, columns=OUTPUT_COLUMNS) if CAL_freq else pd.DataFrame(columns=OUTPUT_COLUMNS)
        return participant_id, df_temp, df_freq
