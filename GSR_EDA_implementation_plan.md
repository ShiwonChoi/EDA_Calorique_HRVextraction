# GSR/EDA implementation plan — main + SoundStress_HRV

Companion to `GSR_EDA_extraction_plan.md` (module design) and
`GSR_EDA_literature_review.md` (citations). This document covers what changes
between branches and what the concrete integration looks like on each.

## Cross-branch applicability

**Yes — this design ports to both `main` and `SoundStress_HRV`, with one
adapted integration layer per branch.** Confirmed by diffing the branches
directly. This repo currently has three branches: `main`, `SoundStress_HRV`,
and `Nested_HRVextraction_archive` (the branch the original plan was written
against) — `Nested_HRVextraction_archive` turns out to be content-identical to
`main`, just nested one directory deeper (`EDA_calorique_HRVextraction/lib/...`
vs `lib/...` at repo root).

What's identical across both studies (verified against a raw file from each:
`Data/SC_03/.../shimmer_P003_Trial00_baseline_...csv` on this branch, and
`/mnt/truenas/Projects/shimmer_pilot_session1_...csv` for Sound-Stress): the
raw shimmer CSV header is byte-identical —
`Time Stamp, GSR (kohms, CAL), Internal ADC A13 (millivolts, CAL)` — same
device/channel config, same unit, same conversion formula. **The entire
`lib/GSR_extract/gsr_preprocess.py` module (conversion, artifact detection,
decomposition, SCR peak detection) and `EDA_temp_extract.py`/`EDA_bin.py`
(metric computation + binning) need zero changes between branches** — they
operate on a raw array + sampling rate + optional `[t_start, t_end]` window,
with no knowledge of trial structure. `EDA_bin.py` will also automatically
inherit whichever `phase_windows`/`label_bin` behavior is correct for each
branch, the same way `HRV_temp_bin.py`'s `bin_temp_30s` is **byte-identical**
between `main` and `SoundStress_HRV` today — only the underlying
`phase_windows` event-label mapping differs per study, and that's imported,
not duplicated.

What differs is the **orchestration layer in `CAL_process.py`**, because the
two branches load/process signals under genuinely different architectures:

- **`main`** (and `Nested_HRVextraction_archive`): per-trial reload.
  `load_and_clean_ppg(participant_path, trial_filter=trial)` is called once
  per trial inside the loop; PPG peak detection and RRI preprocessing each
  re-run per trial.
- **`SoundStress_HRV`**: continuous-recording, compute-once. The whole session
  (baseline + 6 blocks) is one `shimmer_*.csv` + one `event_log_*_aligned.csv`,
  loaded once via `load_and_clean_ppg(participant_path)` (no `trial_filter`).
  Peak detection, RRI preprocessing, and CWT each run **exactly once** on the
  full recording; the per-trial loop then only *masks/windows* the shared
  global results by that trial's `task_window` — explicitly to avoid crashing
  NeuroKit's signal-quality step on short slices and to avoid re-introducing
  filter/wavelet edge artifacts at every trial-cut boundary. This branch
  already established this "compute once, then window per trial via
  `t_start`/`t_end`" pattern for a *second*, independent, non-PPG signal: VAS
  (`lib/Metric_extraction/VAS_extract.py`) loads `touch_data_*.csv` once and
  exposes `get_vas_metrics(df_touch, t_start, t_end)` / `bin_vas_30s(df_touch,
  ...)`, called from inside the per-trial loop exactly the way HRV's
  `get_temp_metrics(..., t_start=, t_end=)` is called. **GSR/EDA is a third
  instance of that exact same established pattern** — not a new architecture,
  just following the branch's own precedent.

Other branch differences that don't affect the GSR design, just the glue code:
- `SoundStress_HRV` has a `requirements.txt` (numpy/pandas/scipy/matplotlib/
  neurokit2>=0.2.13, no `cvxopt`) to extend if the `cvxeda` method is used;
  `main` has no dependency manifest at all — same "add `cvxopt` manually as a
  prerequisite" note applies to both, just nothing to edit on `main`.
- `SoundStress_HRV` inserts a `groupe` column (HC/T) into every output
  dataframe via `group_from_participant()` — `df_gsr` should follow the same
  convention there; not applicable on `main`.
- `SoundStress_HRV`'s `full_process_single` already returns 4 values
  (`participant_id, df_temp, df_freq, df_vas`) and wraps the per-trial body in
  a `try/except` producing `status='FAILED'` rows per trial — `df_gsr` becomes
  a 5th return value there, vs. a 4th on `main`.

**Recommended order of work** (per your request): implement and validate on
`SoundStress_HRV` first — it's arguably the cleaner integration given the
`VAS_extract.py` precedent to mirror almost directly — then port the
(unchanged) core `GSR_extract`/`EDA_*` modules to `main` with only the
`main`-style per-trial-reload integration snippet (already drafted in
`GSR_EDA_extraction_plan.md`) plugged in.

## SoundStress_HRV-specific integration sketch

Mirrors `VAS_extract.py`'s structure almost exactly, except the source is the
continuous `GSR` column already inside `df_ppg` (no separate file to load) and
there's a preprocessing step (conversion + artifact removal + decomposition)
before metrics can be computed.

Once, before the per-trial loop (alongside the existing "5. CWT, ONCE" / "5b.
VAS" steps):
```python
# ── 5c. GSR/EDA — load once ─────────────────────────────────────────────────
eda_uS      = resistance_to_conductance(df_ppg['GSR'])
results_gsr = preprocess_visualize_gsr(df_ppg['time_seconds'], eda_uS, fs,
                                        method=gsr_method, show=show)
```

Inside the per-trial loop, alongside the existing 6a/6b/6c (temporal/freq/VAS)
blocks:
```python
# ── 6d. GSR/EDA — window the shared results_gsr, same shape as get_temp_metrics ──
metrics_gsr = get_eda_metrics(results_gsr, t_start=task_window[0], t_end=task_window[1])
if condition == 'baseline':
    baseline_gsr_raw = metrics_gsr

for metric_name, metric_value in metrics_gsr.items():
    bl = (baseline_gsr_raw or {}).get(metric_name, float('nan'))
    CAL_gsr.extend(build_result_row(
        participant_id=participant_id, trial=trial, condition=condition,
        time_interval_rel_start=0.0, time_interval_abs_start=task_window[0],
        time_interval_rel_end=task_window[1] - task_window[0], time_interval_abs_end=task_window[1],
        task_moment=total_task_moment, recording_type='total',
        metric_name=metric_name, metric_value=metric_value, baseline_mean=bl,
        sample_size=sample_size,
    ))

if condition != 'baseline':
    binned = bin_eda_30s(results_gsr, trial, condition, task_window, df_events_t,
                          participant_id, baseline_gsr_raw or {}, sample_size, bin_width=bin)
    CAL_gsr.extend(binned.to_dict('records'))
```

`full_process_single` returns `(participant_id, df_temp, df_freq, df_vas,
df_gsr)`; `df_gsr` gets the same `groupe` column insert as the other three via
`group_from_participant()`. `batch_process_all` in `main.py` unpacks the
5-tuple and saves a 4th CSV (`..._gsr.csv`).

`SCL_mean`/`SCL_slope` (tonic-derived) and `SCR_count`/`SCR_rate`/
`SCR_amplitude_mean`/`SCR_amplitude_sum`/`AUC_phasic` (phasic-derived) all land
as `Metric` rows in this one `df_gsr` — so tonic and phasic components are
both present but distinguishable by `Metric` name, matching the "one separate
dataframe, using tonic and phasic" request.

## Verification (SoundStress_HRV first pass)

No pilot participant with the flat `SBSA_*`/`SBAA_*` folder layout is present
locally yet — only a loose one-off pilot file was found on the mounted NAS
(`/mnt/truenas/Projects/shimmer_pilot_session1_...csv`, not in the expected
per-participant folder structure `load_ppg` expects). Before running the full
`SoundStress_HRV` batch:
1. Confirm at least one participant folder exists in the shape
   `load_and_clean_ppg` expects (`shimmer_*.csv` + `event_log_*_aligned.csv`
   directly under the participant folder) — either real cohort data or a
   properly-named pilot folder.
2. Run `full_process_single` on that one participant with `show=True`,
   confirm the conductance/artifact-mask/tonic-phasic plots look sane against
   the sound-stress block event markers (block_start/sound_play_start/etc.).
3. Spot-check `df_gsr`'s baseline-referenced rows the same way `df_temp`/`df_vas`
   already are on this branch.
