# Sound Stress — subjective judgement: per-sound VAS extraction

This branch processes the **Sound Stress - subjective judgement** experiment
(2026_Automne): a continuous subjective-stress VAS trace recorded on a touch
slider while 138 sounds are played. There is no HRV and no GSR/EDA in this
branch. It produces **per-sound VAS metrics** (mean, mode, median, std) in a
long-format table.

## Participants

Participant folders are named `JS_##` (e.g. `JS_01`). The folder number is
checked against the `participant=###` entry in the event log (see
[Verification](#verification)); comparison is numeric, so `JS_07` matches `007`
and `JS_10` matches `010`.

## Expected input (per participant folder)

Flat layout, one continuous session per participant (reference: `Data/JS_01`):

- `event_log_*.csv` — event log. Columns: `event_number, event_label,
  utc_timestamp, monotonic_s, elapsed_since_shimmer_s, elapsed_s, block, trial,
  detail`.
- `touch_data_*.csv` — continuous slider recording. Columns: `elapsed_s,
  monotonic_s, position, touch_size, is_touching, timestamp_us`. `position`
  (0–1) is NaN whenever `is_touching == 0`.
- `distress_rating_*.csv/.xlsx` — **ignored**.

Both files carry `monotonic_s`, the clock shared between the event log and the
touch data; it is the only time reference used.

## Setup

Requires Python 3.10+ and the packages in [`requirements.txt`](requirements.txt).
Install with:

```bash
pip install -r requirements.txt
```

> Note: `lib/config.py` still imports `neurokit2`, which this branch does not
> otherwise use. It must be installed (or that import removed) for the pipeline
> to import.

## Running

```bash
python main.py          # batch over all JS_* folders in Data/
```

Single participant, with one plot per sound (opens one window per sound):

```python
from pathlib import Path
from lib.CAL_process import full_process_single

participant_id, df_vas, df_sound = full_process_single(Path("Data/JS_01"), show=True)
```

Output is written to `Results/`:

| File                        | Content                              |
|-----------------------------|--------------------------------------|
| `processed_vas_results.csv` | Per-sound VAS metrics (long format)  |

## Pipeline

`full_process_single(participant_path, show=False)` in
[`lib/CAL_process.py`](lib/CAL_process.py):

1. **Load touch data** — `load_touch_vas` (`lib/Metric_extraction/VAS_extract.py`)
   returns `monotonic_s`, `vas` (= `position × 100`, 0–100 scale) and
   `is_touching`.
2. **Verify and clean** — `load_and_clean_events`
   (`lib/PPG_extract/load_and_clean_events.py`) loads the event log, runs
   `verify_participant`, and cleans events and touch data on one clock.
3. **Match sounds to touch data** — `match_sound_to_touch`
   (`lib/Metric_extraction/match_sound.py`) returns one row per task sound with
   its touch samples.
4. **Per-sound loop** — plots the touch trace when `show=True`
   (`lib/Plots_func/plot_sound.py`), then computes the metrics
   (`lib/Metric_extraction/metric_derive.py`).

Returns `(participant_id, df_vas, df_sound)`.

### Verification

`verify_participant(events, participants_path)` returns two booleans:

- **Participant:** the `detail` of the `experiment_start` row holds
  `participant=###`; its number must equal the trailing number of the folder
  name. A mismatch raises an error.
- **Trials:** `trial` must run 1–138 (`MAX_TASK_TRIALS`), with 138 distinct
  values. A mismatch only prints a warning.

### Cleaning

`load_and_clean_events` keeps the span from `block_start` to `block_end`, and
removes everything between `sound_gap_start`/`sound_gap_end` and
`task_pause_start`/`task_pause_end` (these marker pairs are `EXCLUDE_PAIRS` in
`lib/config.py`). The removed time is subtracted from later timestamps so the
remaining timeline is continuous, and `monotonic_s` is zero-referenced to the
first kept event; times are rounded to the microsecond. Events and touch
samples are cut slightly differently at the gap boundaries:

- **Events:** a gap starts exactly when a sound ends and ends exactly when the
  next one starts, so only events strictly inside a gap or pause are removed;
  the `sound_play_start`/`sound_play_end` events are kept.
- **Touch samples:** cut over `[start, end)`, so a sample at the end of a sound
  is not duplicated at the start of the next one once the gap is collapsed.

Returns `(df_events, df_touch, skips)`, where `skips` lists the removed
intervals (original start, end, duration).

### Sound matching

`match_sound_to_touch` uses `block == 'task'` events only (the 14 training
sounds are excluded). Each sound is a `sound_play_start`/`sound_play_end` pair
with the same `trial`. Returned columns (`df_sound`):

| Column       | Description |
|--------------|-------------|
| `trial`      | `1`–`138`. |
| `category`   | Sound category (part of `detail` before `/`), e.g. `Bruits routiers`. |
| `sound_name` | Sound file (part after `/`), e.g. `Circulation_8.wav`. |
| `start_s`    | Sound start, cleaned zero-referenced `monotonic_s`. |
| `end_s`      | Sound end, same clock. |
| `duration_s` | `end_s - start_s`. |
| `n_samples`  | Touch samples in the interval. |
| `touch`      | DataFrame of the touch samples in `[start_s, end_s)` (`monotonic_s`, `vas`, `is_touching`, `t_rel_s`). |

The last sound's interval is closed at its end.

## Output schema

Long format, one row per `(sound × Metric)`:

| Column           | Description |
|------------------|-------------|
| `participant`    | Folder name, e.g. `JS_01`. |
| `trial`          | `1`–`138`. |
| `category`       | Sound category. |
| `sound_name`     | Sound file name. |
| `interval_start` | Sound start (cleaned `monotonic_s`, seconds). |
| `interval_end`   | Sound end (cleaned `monotonic_s`, seconds). |
| `task_moment`    | Always `task`. |
| `Metric`         | `mean`, `mode`, `median` or `std`. |
| `Value`          | The metric value (float, 0–100 VAS scale). |
| `sample_size`    | Number of touch samples (`is_touching == 1`) in the interval. |
| `status`         | `OK`, `NO_DATA` (no touch samples in the interval) or `FAILED`. |
| `error`          | Error string when `status == FAILED`, else empty. |

138 sounds × 4 metrics = 552 rows per participant.

## Metrics

Computed per sound over the touch samples where `is_touching == 1` (i.e.
`position` is not NaN), on the 0–100 VAS scale
(`get_vas_metrics` in `lib/Metric_extraction/metric_derive.py`):

| `Metric` | Meaning |
|----------|---------|
| `mean`   | Average VAS over the sound. |
| `mode`   | Histogram mode: centre of the most populated bin of width `VAS_MODE_BIN_WIDTH` (1.0 on the 0–100 scale; set in `lib/config.py`). Used because `position` is continuous, so an exact-value mode is not meaningful. Ties go to the lowest bin. |
| `median` | Median VAS over the sound — less sensitive to brief movements than the mean. |
| `std`    | Sample standard deviation (`ddof=1`); NaN with fewer than 2 samples. |

## Plots

With `show=True`, `plot_sound_touch` (`lib/Plots_func/plot_sound.py`) draws, for
each sound as it is processed, the VAS (0–100) against the cleaned
`monotonic_s`, with dashed lines at the sound start (black) and end (red).
Untouched stretches appear as gaps. It opens one blocking window per sound.
