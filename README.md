# Sound Stress — HRV, VAS & GSR/EDA extraction (`SoundStress_HRV`)

This branch processes the **Sound Stress** experiment: one continuous Shimmer
PPG/GSR recording per participant (baseline + 6 stimulus blocks +
post-recovery), plus a continuous subjective-stress VAS trace. It produces
per-trial and 30-second-binned **temporal HRV**, **frequency HRV**, **VAS**,
and **tonic/phasic GSR/EDA** metrics in a shared long-format schema.

## Cohort & groups

Participant folders use two prefixes, mapped to a `groupe` code in every output:

| Folder prefix | `groupe` | Meaning                       |
|---------------|----------|-------------------------------|
| `SBSA_##`     | `HC`     | Healthy controls              |
| `SBAA_##`     | `T`      | Tinnitus group (acouphène)    |

Both prefixes are discovered automatically by the batch runner.

## Expected input (per participant folder)

Flat layout, one continuous session per participant:

- `shimmer_*.csv` — raw Shimmer PPG/GSR (`Internal ADC A13` = PPG, `GSR` = skin
  resistance in kOhm, converted to conductance for EDA analysis).
- `event_log_*_aligned.csv` — event log **aligned to the Shimmer clock** (used).
  The non-`_aligned` `event_log_*.csv` (absolute clock) is **ignored**.
- `touch_data_*.csv` — continuous VAS recording (subjective stress).
- `Processed_PPG/<participant>/…_rr_intervals_corrigé.csv` — corrected RR
  intervals (manually reviewed peaks). Used when present; otherwise auto peaks.
- `distress_rating_*.csv/.xlsx` — **ignored**.

## Setup

Requires Python 3.10+ and the packages in [`requirements.txt`](requirements.txt):
**numpy**, **pandas**, **scipy**, **matplotlib**, **neurokit2** (PPG peak
detection / signal processing / EDA decomposition; pulls in PyWavelets), and
**cvxopt** (only exercised if GSR/EDA is run with `gsr_method='cvxeda'`;
the default `'highpass'` method doesn't need it). Install with:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py          # batch over all SBSA_*/SBAA_* folders
```

Outputs are written to `Results/`:

| File                             | Content                          |
|----------------------------------|----------------------------------|
| `processed_ppg_results_temp.csv` | Temporal HRV                     |
| `processed_ppg_results_freq.csv` | Frequency HRV (band power)       |
| `processed_vas_results.csv`      | VAS subjective-stress statistics |
| `processed_gsr_results.csv`      | Tonic/phasic GSR/EDA statistics  |

## Output schema (shared by all four CSVs)

All four files are **long format** with one row per
`(trial × condition × Metric × Value_type × recording window)`:

| Column                      | Description |
|-----------------------------|-------------|
| `participant`               | Folder name, e.g. `SBSA_02`. |
| `groupe`                    | `HC` (SBSA control) or `T` (SBAA tinnitus). |
| `trial`                     | `0` = baseline; `1`–`6` = stimulus blocks. |
| `condition`                 | `baseline`, or `<block>_<design>` (see below). |
| `time_interval_rel_start`   | Window start, seconds relative to the trial's task start. |
| `time_interval_abs_start`   | Window start, seconds since Shimmer connection. |
| `time_interval_rel_end`     | Window end, relative to trial task start. |
| `time_interval_abs_end`     | Window end, seconds since Shimmer connection. |
| `task_moment`               | Recording phase (see below). |
| `recording_type`            | `total` (whole trial) or `interval` (a 30 s bin). |
| `Metric`                    | The metric name (see per-file tables). |
| `Value_type`                | `raw`, `diff`, `pct_change`, or `log_ratio`. |
| `Value`                     | The numeric value. |
| `sample_size`               | See per-file notes. |
| `status`                    | `SUCCESS` or `FAILED`. |
| `error`                     | Error string when `status == FAILED`, else empty. |

### `condition` values

`baseline` for trial 0. For blocks 1–6, `condition = <block label>_<design>`,
combining the sound label with the stimulus design, e.g.
`quiet_individu`, `loud_individu`, `original_quatre_sons`,
`loud_quatre_sons`, `quiet_quatre_sons`, `original_individu`.

### `recording_type` and `task_moment`

- `total` rows cover the whole trial (first→last event). `task_moment` is
  `baseline` for trial 0, otherwise `total`.
- `interval` rows are sequential **30 s bins** (last bin may be shorter),
  emitted for stimulus blocks only (baseline is kept whole). Each bin's
  `task_moment` is the phase whose window contains the bin centre:
  `anticipation` (block_start→countdown_start), `task`
  (sound_play_start→sound_play_end), `recovery` (rest, or post_recovery for
  block 6), or `unclassified`.
- **VAS, HRV, and GSR bins share identical bin edges**, so rows can be joined
  on `trial` + `time_interval_abs_start`/`_end`.

### `Value_type` (baseline referencing)

Each metric appears in four rows:

- `raw` — the metric value itself.
- `diff` — value minus the baseline reference.
- `pct_change` — percent change vs baseline.
- `log_ratio` — `ln(value / baseline)`.

Baseline (trial 0) rows carry `raw` and set `diff`/`pct_change`/`log_ratio` to
`0.0` by convention. Temporal HRV, VAS, and GSR are referenced against the
baseline **trial's** whole-trial value; frequency HRV is referenced **per
frequency** against the baseline window of the shared wavelet transform.

## Per-file metrics

### `processed_ppg_results_temp.csv` — temporal HRV

Computed directly from the beat-to-beat (RR) interval series. These indicate
**overall autonomic activity on the heart**, mixing sympathetic ("fight or
flight") and parasympathetic/vagal ("rest and digest") influence — RMSSD in
particular is the standard proxy for vagal/parasympathetic tone.

| `Metric`   | Meaning |
|------------|---------|
| `mean_HR`  | Average heart rate (beats per minute) over the window. |
| `mean_RRI` | Average time between heartbeats (ms) — the inverse of `mean_HR`. |
| `RMSSD`    | Root-mean-square of successive beat-to-beat differences (ms) — short-term variability; higher generally reflects **more vagal/parasympathetic activity** (calmer state). |
| `SDNN`     | Standard deviation of all beat intervals (ms) — overall HRV across the whole window, mixing both branches of the autonomic nervous system. |

`sample_size` = `"<clean beats> / <raw beats>"` in the window.

### `processed_ppg_results_freq.csv` — frequency HRV

Band power from a continuous wavelet transform of the same RR-interval series
(Task Force 1996 band definitions). These decompose HRV by oscillation speed,
traditionally interpreted as separating slower sympathetic-linked rhythms from
faster vagal/parasympathetic-linked rhythms — an interpretation the HRV
literature treats as a useful heuristic rather than a strict law, since LF in
particular is now known to reflect a mix of both branches.

| `Metric` | Meaning |
|----------|---------|
| `VLF`    | Very-low-frequency power (0.003–0.04 Hz) — slow regulatory rhythms (e.g. thermoregulation, hormonal). |
| `LF`     | Low-frequency power (0.04–0.15 Hz) — historically linked to sympathetic activity and baroreflex, but now considered a mixed sympathetic/vagal signal. |
| `HF`     | High-frequency power (0.15–0.40 Hz) — tracks respiration-linked ("respiratory sinus arrhythmia") vagal/parasympathetic activity. |

`sample_size` = `"<clean beats> / <raw beats>"`.

### `processed_vas_results.csv` — subjective stress (VAS)

The participant's own **self-reported, moment-to-moment stress rating**
(0–100 slider), independent of any physiological signal.

| `Metric`     | Meaning |
|--------------|---------|
| `VAS_mean`   | Average self-reported stress level over the window. |
| `VAS_median` | Median self-reported stress level — less sensitive to brief spikes than the mean. |
| `VAS_std`    | Variability of self-reported stress within the window. |

`sample_size` = number of touch samples in the window.

**VAS score** = `position × 100` (0–100 scale). The VAS clock (`elapsed_s`,
zeroed to VAS-recording start) is shifted onto the Shimmer timeline via
`elapsed_s + touchslider_recording_start` (from the aligned event log), so VAS
windows match the HRV windows exactly. The `touch_size`, `is_touching`, and
`timestamp_us` columns of `touch_data` are not used. If a participant has no
`touch_data`/marker, VAS is skipped (empty rows) without failing HRV.

### `processed_gsr_results.csv` — tonic/phasic GSR/EDA

Computed from the Shimmer `GSR` channel (skin resistance, kOhm → converted to
conductance, µS) after artifact removal, then split into a slow **tonic**
component and a fast **phasic** component (`gsr_method`, default `'highpass'`).
Both reflect **sweat-gland activity driven by the sympathetic nervous system
only** — unlike HRV, EDA has no parasympathetic/vagal contribution, so it's a
comparatively unambiguous arousal signal.

**Tonic (skin conductance level, SCL)** — the slow-moving background level,
reflecting sustained/overall sympathetic arousal (e.g. general stress level
during a block, independent of any single moment):

| `Metric`          | Meaning |
|-------------------|---------|
| `Tonic_SCL_mean`   | Average background skin conductance (µS) over the window — higher means more sustained sympathetic arousal. |
| `Tonic_SCL_slope`  | Linear trend of the background level over the window (µS/s) — rising = building arousal/sensitization, falling = habituation/recovery. |

**Phasic (skin conductance responses, SCRs)** — brief spikes on top of the
tonic level, each one a discrete sympathetic "startle"/arousal event, either
triggered by a specific stimulus or occurring spontaneously:

| `Metric`                    | Meaning |
|-----------------------------|---------|
| `Phasic_SCR_count`          | Number of distinct skin-conductance-response spikes detected in the window. |
| `Phasic_SCR_rate`           | Same count, normalized to responses per minute — comparable across windows of different length. |
| `Phasic_SCR_amplitude_mean` | Average size (µS) of the detected responses — how strong each arousal spike is, on average. |
| `Phasic_SCR_amplitude_sum`  | Total size (µS) of all detected responses added together — combined intensity of arousal in the window. |
| `Phasic_AUC`                | Area under the phasic curve (µS·s) over the window — a single combined measure of both how many responses occurred and how large they were. |

`sample_size` = `"<clean samples> / <raw samples>"` in the window (same
convention as temporal HRV, but counting GSR samples instead of heartbeats).
