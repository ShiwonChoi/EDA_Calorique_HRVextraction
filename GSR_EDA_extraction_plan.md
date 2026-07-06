# GSR/EDA extraction for the caloric-stress study — implementation plan

## Context

The Shimmer devices record both PPG and GSR (skin resistance, kOhms) in the same
`shimmer_*.csv` per trial, but only the PPG channel is currently extracted into
HRV metrics (`lib/CAL_process.py::full_process_single`). The GSR column already
flows through the shared per-trial loader (`load_and_clean_ppg` → `resample_signal`,
which resamples *all* numeric columns, not just PPG) but is discarded downstream.
This plan adds a parallel electrodermal-activity (EDA) pipeline — preprocessing,
tonic/phasic decomposition, and total/binned feature extraction — architected to
mirror the existing HRV pipeline's structure and output schema, and wires it into
`full_process_single` as a third output alongside `df_temp`/`df_freq`.

Two concrete data problems were confirmed by inspecting a raw file
(`Data/SC_03/Stress measures/shimmer_P003_Trial00_baseline_20260205_143554.csv`):
the native GSR sampling rate is **64 Hz** (not the 250 Hz the shared resampler
produces), and the raw "CAL" kOhm column contains physiologically impossible
negative values (e.g. -983 kOhm) and runs of exactly-repeated values —
consistent with Shimmer's auto-ranging hardware behavior (documented in
Shimmer's own technical guide: range changes hold/duplicate the last value
through an ~80 ms settling window). These must be handled before any
decomposition, not just visualized.

See `GSR_EDA_literature_review.md` (companion document, same directory) for the
full peer-review-grounded literature survey backing the decisions below,
including honesty flags on what could and could not be verified.

Key findings from that review, summarized:

- 64 Hz is adequate for tonic/SCL work and non-specific SCR counting but below
  the ≥200 Hz some guidance recommends for precise SCR rise-time/latency work
  (SPR committee report — Boucsein et al. 2012, *Psychophysiology* 49(8);
  Boucsein, *Electrodermal Activity*, 2nd ed., 2012). Upsampling via
  interpolation (already done at 250 Hz alongside PPG) is a neutral
  resampling-for-alignment step — it does **not** manufacture new temporal
  resolution, so metrics relying on precise SCR timing should be read with that
  caveat.
- No peer-reviewed paper specifically documenting Shimmer3 GSR+ range-switching
  artifacts could be verified — the artifact-handling approach below is
  grounded in Shimmer's own technical documentation plus general EDA
  data-quality literature (Kleckner et al. 2018; Taylor et al. 2015 EDA
  Explorer), not a Shimmer-specific paper.
- Conductance (µS), not resistance, is the near-universal analysis domain
  (Dawson, Schell & Filion, *Handbook of Psychophysiology* 4th ed., 2016 ch. 10;
  Boucsein 2012). `Conductance_µS = 1000 / Resistance_kOhm`.
- Decomposition: cvxEDA (Greco et al. 2016) and Ledalab's CDA (Benedek &
  Kaernbach 2010, *J Neurosci Methods*) are the two most-cited model-based
  methods in current practice. CDA is only available via the MATLAB Ledalab
  toolbox (no maintained Python port) — out of scope for now, flagged as future
  work. cvxEDA and the simple Butterworth "highpass" method (Biopac/AcqKnowledge
  convention, no independent peer-reviewed origin, but this codebase's existing
  style — see `RRI_preprocess.py`'s own high-pass detrending) are both available
  in NeuroKit2 and will be implemented as swappable options. Sparse deconvolution
  (Hernando-Gallego et al. 2018) is available in NeuroKit2 but flagged by
  NeuroKit2 itself as experimental — included as a third, clearly-labeled option.
- SCR/threshold conventions split between a legacy 0.05 µS minimum and a modern
  0.01 µS minimum for higher-resolution digital systems (SPR guidance; the
  Braithwaite et al. 2013 Birmingham SAAL lab guide — note: this is a technical
  report, not a peer-reviewed journal article, and is frequently mis-cited with
  the SPR paper's own journal citation). Standard reported metrics: SCL mean,
  SCL slope, SCR rate/frequency, SCR amplitude mean/sum, area under the phasic
  curve, non-specific-SCR rate.

## Design decisions (confirmed with user)

1. **Signal source**: reuse the GSR column already resampled to 250 Hz inside
   `df_ppg_t` by the existing `load_and_clean_ppg` call in
   `full_process_single` — no new file-parsing/resampling code.
2. **Decomposition**: implement as a swappable `method` parameter (default
   `'highpass'`), exposing NeuroKit2's `'highpass'` and `'cvxeda'` as the two
   primary supported methods (the two most literature-relevant, Python-available
   options), with `'sparse'` available but documented as experimental. A single
   pipeline-wide parameter (not per-trial/per-condition branching) satisfies
   "choice based on a condition" without speculative complexity — nothing in
   the data suggests different trials need different decomposition methods.
3. **SCR correction**: fully automatic (`nk.eda_peaks`) for this first pass, no
   manual-correction workflow (unlike PPG's `manual_peak.py`).
4. **Output**: a third dataframe `df_gsr`, same `OUTPUT_COLUMNS` schema, no
   changes needed to `config.py`'s schema (already metric-agnostic).

## New modules

**`lib/GSR_extract/gsr_preprocess.py`** (mirrors `PPG_extract/ppg_preprocess.py` +
`Metric_extraction/RRI_preprocess.py`):
- `resistance_to_conductance(gsr_kohm) -> eda_uS`: `1000 / gsr_kohm`.
- `detect_artifacts_gsr(eda_uS, plausible_range=(0.05, 60.0), max_slope=10.0)`:
  mirrors `RRI_preprocess.detect_artifacts`'s two-tier pattern — physiological
  bounds (Kleckner et al. 2018's 0.05–60 µS range, catches the negative-value
  artifact directly) plus a max-slope check (≈10 µS/s, catches the abrupt
  jumps at range-switch boundaries). Returns a boolean artifact mask.
- `remove_artifacts_gsr(eda_uS, artifact_mask, method='interpolate')`: reuses
  the same remove/interpolate pattern as `RRI_preprocess.remove_artifacts`.
- `decompose_eda(eda_clean, fs, method='highpass') -> {'tonic': ..., 'phasic': ...}`:
  thin wrapper around `nk.eda_phasic(eda_clean, sampling_rate=fs, method=method)`.
- `detect_scr_peaks(phasic, fs, amplitude_min=SCR_AMPLITUDE_MIN_US)`: wrapper
  around `nk.eda_peaks`, returns SCR onsets/peaks/amplitudes.
- `preprocess_visualize_gsr(time_s, gsr_kohm, fs, method='highpass', ...) -> results dict`:
  orchestrator mirroring `RRI_preprocess.preprocess_visualize`'s `results` dict
  shape (raw/clean/artifact_mask/tonic/phasic/scr_info/params), used identically
  by the binning functions and the plotting function.
- `plot_preprocessing_steps_gsr(results, participant_id, df_events, trial, show)`:
  mirrors `plot_preprocessing_steps` — raw-with-artifacts, cleaned, tonic vs.
  phasic decomposition, with event vlines.

**`lib/Metric_extraction/EDA_temp_extract.py`** (mirrors `HRV_temp_extract.py`):
- `get_eda_metrics(results_gsr, t_start=None, t_end=None) -> dict`: computes,
  optionally windowed to `[t_start, t_end]`:
  `SCL_mean`, `SCL_slope` (linear-regression slope of tonic over the window,
  µS/s — habituation trend), `SCR_count`, `SCR_rate` (per minute), `SCR_amplitude_mean`,
  `SCR_amplitude_sum`, `AUC_phasic` (trapezoidal integral, µS·s).
- Directly imports and reuses `phase_windows` / `label_bin` from
  `HRV_temp_extract.py` (already signal-agnostic) — no duplication.

**`lib/Metric_extraction/EDA_bin.py`** (mirrors `HRV_temp_bin.py`):
- `bin_eda_total(results_gsr, trial, condition, task_interval, participant_id, sample_size)`:
  one whole-trial row per metric, `recording_type='total'`, baseline-referenced
  against Trial00 via the existing `build_result_row` (unchanged, already
  generic on metric_name/value/baseline_mean/condition).
- `bin_eda_30s(results_gsr, ..., bin_width=30)`: sequential fixed-width bins,
  `recording_type='interval'`, stim trials only — same structure as
  `bin_temp_30s`.

**`config.py`** additions (new constants block, no schema changes):
- `GSR_PLAUSIBLE_RANGE_US = (0.05, 60.0)`, `GSR_MAX_SLOPE_US_PER_S = 10.0`
  (Kleckner et al. 2018), `SCR_AMPLITUDE_MIN_US = 0.05` (conservative SPR-era
  default; document the 0.01 µS modern alternative in a comment, adjustable).

## `CAL_process.py` integration

Inside `full_process_single`'s existing per-trial loop, add a step "6c. GSR/EDA"
after the existing 6a (temporal) / 6b (frequency) blocks, using `df_ppg_t`
(already loaded in step 1 — no new load call) and `fs`:

```python
eda_uS = resistance_to_conductance(df_ppg_t['GSR'])
results_gsr = preprocess_visualize_gsr(df_ppg_t['time_seconds'], eda_uS, fs, method=gsr_method)

if condition == 'baseline':
    baseline_gsr_raw = get_eda_metrics(results_gsr, *task_window)

total = bin_eda_total(results_gsr, trial, condition, task_window, participant_id, sample_size)
CAL_gsr.extend(total.to_dict('records'))   # via build_result_row internally, same as CAL_temp

if condition != 'baseline':
    binned = bin_eda_30s(results_gsr, trial, condition, task_window, df_events_t,
                          participant_id, baseline_gsr_raw, sample_size, bin_width=bin)
    CAL_gsr.extend(binned.to_dict('records'))

if show:
    plot_preprocessing_steps_gsr(results_gsr, participant_id, df_events_t, trial, show=show)
```

`full_process_single` signature gains `gsr_method='highpass'` (new keyword,
default preserves existing call sites) and returns
`(participant_id, df_temp, df_freq, df_gsr)`.

## `main.py` integration

- `batch_process_all` unpacks the 4-tuple, accumulates `all_gsr`, concats to
  `df_gsr_all`, returns 3 dataframes.
- `__main__` block saves a third CSV:
  `gsr_out = output_file.with_name(output_file.stem + "_gsr.csv")`, mirroring
  the existing `_temp.csv`/`_freq.csv` pattern exactly.

## Prerequisite

`cvxeda` decomposition method requires the `cvxopt` package (NeuroKit2's
dependency for that method) — no `requirements.txt` exists in this repo
currently, so this needs a manual `pip install cvxopt` before that method path
is exercised; `'highpass'` (the default) has no new dependencies.

## Verification

No automated tests exist in this repo currently (confirmed) — validation
follows the same pattern already used for the HRV pipeline:
1. Run `full_process_single` on one participant with `show=True` (start with
   `SC_03`, where the raw-signal artifacts were directly observed) and visually
   confirm: conductance values are physiologically plausible (no negative
   values survive), the artifact mask catches the range-switch
   plateaus/negative spikes, and the tonic/phasic split looks sane relative to
   event markers (SCRs should cluster near stimulation/vibration onsets).
2. Spot-check `df_gsr`'s baseline-referenced `diff`/`pct_change`/`log_ratio`
   rows against the same convention already validated for `df_temp`.
3. Only after visual validation on 1-2 participants, run the full
   `batch_process_all` across the cohort.
