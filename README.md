# Sound Stress — HRV & VAS extraction (`SoundStress_HRV`)

This branch processes the **Sound Stress** experiment: one continuous Shimmer
PPG/GSR recording per participant (baseline + 6 stimulus blocks +
post-recovery), plus a continuous subjective-stress VAS trace. It produces
per-trial and 30-second-binned **temporal HRV**, **frequency HRV**, and **VAS**
metrics in a shared long-format schema.

## Cohort & groups

Participant folders use two prefixes, mapped to a `groupe` code in every output:

| Folder prefix | `groupe` | Meaning                       |
|---------------|----------|-------------------------------|
| `SBSA_##`     | `HC`     | Healthy controls              |
| `SBAA_##`     | `T`      | Tinnitus group (acouphène)    |

Both prefixes are discovered automatically by the batch runner.

## Expected input (per participant folder)

Flat layout, one continuous session per participant:

- `shimmer_*.csv` — raw Shimmer PPG/GSR (`Internal ADC A13` = PPG).
- `event_log_*_aligned.csv` — event log **aligned to the Shimmer clock** (used).
  The non-`_aligned` `event_log_*.csv` (absolute clock) is **ignored**.
- `touch_data_*.csv` — continuous VAS recording (subjective stress).
- `Processed_PPG/<participant>/…_rr_intervals_corrigé.csv` — corrected RR
  intervals (manually reviewed peaks). Used when present; otherwise auto peaks.
- `distress_rating_*.csv/.xlsx` — **ignored**.

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

## Output schema (shared by all three CSVs)

All three files are **long format** with one row per
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
- **VAS and HRV bins share identical bin edges**, so rows can be joined on
  `trial` + `time_interval_abs_start`/`_end`.

### `Value_type` (baseline referencing)

Each metric appears in four rows:

- `raw` — the metric value itself.
- `diff` — value minus the baseline reference.
- `pct_change` — percent change vs baseline.
- `log_ratio` — `ln(value / baseline)`.

Baseline (trial 0) rows carry `raw` and set `diff`/`pct_change`/`log_ratio` to
`0.0` by convention. Temporal HRV and VAS are referenced against the baseline
**trial's** whole-trial value; frequency HRV is referenced **per frequency**
against the baseline window of the shared wavelet transform.

## Per-file metrics

### `processed_ppg_results_temp.csv` — temporal HRV
`Metric` ∈ `mean_HR`, `mean_RRI`, `RMSSD`, `SDNN`.
`sample_size` = `"<clean beats> / <raw beats>"` in the window.

### `processed_ppg_results_freq.csv` — frequency HRV
`Metric` ∈ `VLF`, `LF`, `HF` (band power from a continuous wavelet transform;
Task Force 1996 bands). `sample_size` = `"<clean beats> / <raw beats>"`.

### `processed_vas_results.csv` — subjective stress (VAS)
`Metric` ∈ `VAS_mean`, `VAS_median`, `VAS_std` of the VAS score.
`sample_size` = number of touch samples in the window.

**VAS score** = `position × 100` (0–100 scale). The VAS clock (`elapsed_s`,
zeroed to VAS-recording start) is shifted onto the Shimmer timeline via
`elapsed_s + touchslider_recording_start` (from the aligned event log), so VAS
windows match the HRV windows exactly. The `touch_size`, `is_touching`, and
`timestamp_us` columns of `touch_data` are not used. If a participant has no
`touch_data`/marker, VAS is skipped (empty rows) without failing HRV.
