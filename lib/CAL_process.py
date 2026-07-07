import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg
from lib.PPG_extract.manual_peak import (
    load_corrected_rri_for_participant, detect_peaks_full, place_corrected_peaks,
    extract_beats, plot_corr_peaks,
)
from lib.Metric_extraction.RRI_preprocess import preprocess_pipeline, preprocess_visualize, plot_preprocessing_steps, validate_intervals
from lib.Metric_extraction.HRV_temp_extract import get_temp_metrics
from lib.Metric_extraction.HRV_temp_bin import bin_temp_30s
from lib.Metric_extraction.HRV_freq_extract import (
    run_cwt_compute, run_cwt_task, extract_band_power_cwt, band_results_to_df, plot_cwt_scalogram,
)
from lib.Metric_extraction.HRV_freq_bin import bin_totalbandpower, bin_bandpower_30s
from lib.Metric_extraction.HRV_df import build_result_row
from lib.Metric_extraction.VAS_extract import (
    get_vas_recording_offset, load_touch_vas, get_vas_metrics, bin_vas_30s,
)
from lib.config import OUTPUT_COLUMNS, group_from_participant


def _masked_count(beat_times, t_start, t_end):
    """Count of beat times falling inside [t_start, t_end]."""
    t = np.asarray(beat_times, dtype=float)
    if t.size == 0:
        return 0
    return int(np.sum((t >= t_start) & (t <= t_end)))


def full_process_single(participant_path, use_physio=True, use_stat=False, show=False, bin=30):
    """
    Process one participant's continuous PPG recording.

    The whole session (baseline + 6 stimulus blocks) is one continuous
    recording, loaded once via load_and_clean_ppg. Peak detection, RRI
    extraction/preprocessing, and CWT computation each run exactly ONCE on
    the full recording — not per trial, since re-running them on a short
    trial slice (e.g. a brief baseline window) can crash (too few peaks for
    NeuroKit's signal-quality step) and re-introduces filter/wavelet edge
    artifacts at every trial-cut boundary instead of only at the true
    recording start/end.

    Trials are the 7 windows derived by assign_trial_condition:
        0        'baseline'
        1..6     '<sound>_<design>' e.g. 'loud_individu', 'quiet_quatre_sons'
    Block 6 has no rest_start/rest_end of its own, so its window (and
    'recovery' phase) extends through post_recovery_start/post_recovery_end
    instead — see phase_windows in HRV_temp_extract.py.

    Per trial, only masking/labeling/binning happens — each trial's metrics
    are computed by time-windowing the ONE global preprocessed RRI array /
    CWT matrix to that trial's task_window, via get_temp_metrics/bin_temp_30s's
    existing t_start/t_end masking and run_cwt_task's existing task_window
    masking. A failure on one trial (status="FAILED" row) doesn't discard the
    other trials' results.

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

    CAL_vas collects subjective-stress VAS rows (VAS_mean, VAS_median, VAS_std)
    from touch_data_*.csv, mapped onto the shimmer-connected timeline and
    binned over the SAME task windows / 30-s bins as CAL_temp, baseline-
    referenced the same way. Skipped (empty) if no touch_data / recording
    marker is present.

    Returns:
        participant_id (str)         : Participant identifier (e.g. "SBSA_02").
        df_temp        (pd.DataFrame): Temporal HRV rows, schema = OUTPUT_COLUMNS.
        df_freq        (pd.DataFrame): Frequency HRV rows, schema = OUTPUT_COLUMNS.
        df_vas         (pd.DataFrame): VAS rows, schema = OUTPUT_COLUMNS.
    """
    participant_path = Path(participant_path)
    print(f"\nParticipant folder : {participant_path.name}")

    CAL_temp              = []
    CAL_freq              = []
    CAL_vas               = []
    baseline_temp_raw     = None
    baseline_per_freq_raw = None
    baseline_vas_raw      = None
    participant_id        = None

    try:
        # ── 1. Load the whole continuous session once ──────────────────────
        df_ppg, df_events, fs, _, participant_id = load_and_clean_ppg(participant_path, show=show)
        trials = sorted(df_events['trial'].dropna().unique())
        print(f"Trials found       : {trials}")

        # ── 2. Peak detection + corrected-peak placement, ONCE ─────────────
        signal, info = detect_peaks_full(df_ppg, fs)

        df_corr, base_rri = load_corrected_rri_for_participant(
            participant_path, participant_id, use_physio=use_physio, use_stat=use_stat,
        )
        signal = place_corrected_peaks(signal, df_ppg, df_corr, use_physio=use_physio, use_stat=use_stat)

        # Attach time/metadata so `signal` is self-contained
        signal["time_seconds"]      = df_ppg["time_seconds"].values
        signal["rel_zero_ref (ms)"] = df_ppg["rel_zero_ref (ms)"].values
        signal["abs_zero_ref (ms)"] = df_ppg["abs_zero_ref (ms)"].values
        signal["participant"]       = participant_id

        if show:
            plot_corr_peaks(signal, df_ppg, df_events, df_corr, participant_id, show=show)
        print(f"  Full recording peaks placed — signal shape: {signal.shape}")

        # ── 3. Beat/RRI extraction, ONCE ────────────────────────────────────
        beat_times, rri_values = extract_beats(signal, df_ppg)
        print(f"  RRI created: {np.sum(~np.isnan(rri_values))} valid samples across full recording")

        # ── 4. RRI preprocessing, ONCE ───────────────────────────────────────
        results = preprocess_visualize(beat_times, rri_values, use_physio=use_physio, use_stat=use_stat)

        print("\nCreating tachogram from raw RRI to filtered RRI")
        plot_preprocessing_steps(results, participant_id=participant_id, show=show)

        print("\nPreprocessing complete.")
        print(f"  Input:  {len(results['intervals_raw'])} beats (irregular)")
        print(f"  Output: {len(results['intervals_clean'])} beats cleaned")           # Temporal HRV
        print(f"  Output: {len(results['intervals_resampled'])} resampled beats")     # Cubic spline interpolation
        print(f"  Output: {len(results['intervals_filtered'])} samples at {results['fs_resample']} Hz (uniform, filtered)")  # Frequency HRV

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

        # ── 5. CWT, ONCE for the whole recording ────────────────────────────
        print("\n  Computing CWT for full recording...")
        cwt_r = run_cwt_compute(results['intervals_filtered'], results['t_resampled'],
                                results['fs_resample'], verbose=False)

        # ── 5b. VAS (subjective stress) — load once, optional ───────────────
        # touch_data_*.csv is independent of PPG; a missing file/marker just
        # skips VAS extraction without failing the participant.
        vas_offset = get_vas_recording_offset(participant_path)
        df_touch   = load_touch_vas(participant_path, vas_offset)
        if df_touch is None:
            print("  No VAS/touch data — VAS extraction skipped for this participant")

        # ── 6. Per-trial loop: masking / labeling / binning only ────────────
        for trial in trials:
            print(f"\n{'─' * 55}")
            print(f"  Trial: {trial}")
            print(f"{'─' * 55}")

            try:
                df_events_t = df_events[df_events['trial'] == trial].reset_index(drop=True)
                condition   = df_events_t['condition'].iloc[0]

                # Task window: first → last event for this trial (seconds since device connected)
                ev_times_s  = df_events_t['time_since_connected_ms'].values / 1000
                task_window = (float(ev_times_s[0]), float(ev_times_s[-1]))

                n_clean = _masked_count(results['beat_times_clean'], *task_window)
                n_raw   = _masked_count(results['beat_times_raw'],   *task_window)
                sample_size = f"{n_clean} / {n_raw}"

                # task_moment for whole-trial (total) rows
                total_task_moment = 'baseline' if condition == 'baseline' else 'total'

                # -- 6a. Temporal ---------------------------------------------------------------
                metrics_temp = get_temp_metrics(
                    results['intervals_clean'], results['beat_times_clean'],
                    t_start=task_window[0], t_end=task_window[1],
                )
                if condition == 'baseline':
                    baseline_temp_raw = metrics_temp

                for metric_name, metric_value in metrics_temp.items():
                    bl = (baseline_temp_raw or {}).get(metric_name, float('nan'))
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
                        baseline_temp_raw or {}, sample_size, bin_width=bin,
                    )
                    CAL_temp.extend(temp_binned.to_dict('records'))

                # ── 6b. Frequency (CWT band power) ────────────────
                if condition == 'baseline':
                    # Trial 0 IS the baseline — raw band power only, no
                    # per-frequency correction to run against itself. Its own
                    # per-frequency mean (over its own window of the shared
                    # cwt_r) becomes the reference other trials are corrected
                    # against below.
                    band_results = extract_band_power_cwt(cwt_r)
                    band_df = band_results_to_df(
                        band_results, participant_id=participant_id,
                        drop_outside_task_window=False,
                    )

                    bl_mask = (cwt_r['times'] >= task_window[0]) & (cwt_r['times'] < task_window[1])
                    baseline_per_freq_raw = np.nanmean(cwt_r['power'][:, bl_mask], axis=1)
                else:
                    # Per-frequency correction against trial 0's baseline_per_freq_raw,
                    # masking the SAME shared cwt_r to this trial's own task_window.
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

                # ── 6c. VAS (subjective stress) ────────────────────────────
                # Same task window / 30-s bins / baseline reference as the
                # temporal HRV metrics above.
                if df_touch is not None:
                    vas_metrics, vas_n = get_vas_metrics(df_touch, task_window[0], task_window[1])
                    if condition == 'baseline':
                        baseline_vas_raw = vas_metrics

                    for metric_name, metric_value in vas_metrics.items():
                        bl = (baseline_vas_raw or {}).get(metric_name, float('nan'))
                        CAL_vas.extend(build_result_row(
                            participant_id=participant_id,
                            trial=trial,
                            condition=condition,
                            time_interval_rel_start=0.0,
                            time_interval_abs_start=task_window[0],
                            time_interval_rel_end=task_window[1] - task_window[0],
                            time_interval_abs_end=task_window[1],
                            task_moment=total_task_moment, recording_type='total',
                            metric_name=metric_name, metric_value=metric_value,
                            baseline_mean=bl, sample_size=f"{vas_n}",
                        ))

                    # 30-s binned VAS — stim trials only (baseline kept whole)
                    if condition != 'baseline':
                        vas_binned = bin_vas_30s(
                            df_touch, trial, condition, task_window, df_events_t,
                            participant_id, baseline_vas_raw or {}, bin_width=bin,
                        )
                        CAL_vas.extend(vas_binned.to_dict('records'))

                    print("  VAS total: " + "  ".join(
                        f"{k}={v:.2f}" for k, v in vas_metrics.items()
                    ))

            except Exception as trial_err:
                import traceback
                print(f"\n  ERROR processing trial {trial}: {trial_err}")
                traceback.print_exc()
                failed_row = {
                    'participant': participant_id, 'trial': trial,
                    'status': 'FAILED', 'error': str(trial_err),
                }
                CAL_temp.append(failed_row)
                CAL_freq.append(failed_row)
                CAL_vas.append(failed_row)

        # Output dataframes for HRV temp/freq and VAS metrics
        df_temp = pd.DataFrame(CAL_temp, columns=OUTPUT_COLUMNS) if CAL_temp else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_freq = pd.DataFrame(CAL_freq, columns=OUTPUT_COLUMNS) if CAL_freq else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_vas  = pd.DataFrame(CAL_vas,  columns=OUTPUT_COLUMNS) if CAL_vas  else pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Study-group tag (HC = SBSA controls, T = SBAA tinnitus), placed
        # right after the participant column.
        groupe = group_from_participant(participant_id)
        for _df in (df_temp, df_freq, df_vas):
            _df.insert(1, 'groupe', groupe)

        print(f"\n{'=' * 55}")
        print(f"  All trials processed for {participant_id}.")
        print(f"  CAL_temp rows : {len(df_temp)}")
        print(f"  CAL_freq rows : {len(df_freq)}")
        print(f"  CAL_vas  rows : {len(df_vas)}")
        print(f"{'=' * 55}")

        return participant_id, df_temp, df_freq, df_vas

    except Exception as e:
        import traceback
        print(f"\nERROR processing {participant_path.name}: {e}")
        traceback.print_exc()
        df_temp = pd.DataFrame(CAL_temp, columns=OUTPUT_COLUMNS) if CAL_temp else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_freq = pd.DataFrame(CAL_freq, columns=OUTPUT_COLUMNS) if CAL_freq else pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_vas  = pd.DataFrame(CAL_vas,  columns=OUTPUT_COLUMNS) if CAL_vas  else pd.DataFrame(columns=OUTPUT_COLUMNS)
        groupe = group_from_participant(participant_id)
        for _df in (df_temp, df_freq, df_vas):
            _df.insert(1, 'groupe', groupe)
        return participant_id, df_temp, df_freq, df_vas
