# Sound Stress — subjective judgement: per-sound VAS extraction

This branch processes the **Sound Stress - subjective judgement** experiment
(2026_Automne). During the session, 138 sounds are played while participants
rate their subjective stress continuously on a touch slider (VAS). There is no
HRV and no GSR/EDA in this branch.

A typical judgement of one sound has three phases:

1. **Delay**: no judgement yet, while the slider stays near 0.
2. **Adjustment**: the slider rises to the judgement, sometimes with an
   overshoot or undershoot.
3. **Plateau**: the final judgement, held for the rest of the sound.

The pipeline extracts **three-phase metrics** for each sound and writes them in
a long-format table. The literature behind these metrics is summarised in
[`VAS_metrics_literature.md`](VAS_metrics_literature.md).

> ⚠️ **Read [Limitations](#limitations) before using the metrics.** They are
> reliable for sounds that follow the typical delay → adjustment → plateau
> pattern, and less so for very short sounds, sounds with more than two
> plateaus, or atypical judgements.

## Participants

Participant folders are named `JS_##` (e.g. `JS_01`). The folder number is
checked against the `participant=###` entry in the event log (see
[Verification](#verification)). The comparison is numeric, so `JS_07` matches
`007` and `JS_10` matches `010`.

## Expected input (per participant folder)

Each participant folder has a flat layout with one continuous session
(reference: `Data/JS_01`):

- `event_log_*.csv` is the event log. Columns: `event_number, event_label,
  utc_timestamp, monotonic_s, elapsed_since_shimmer_s, elapsed_s, block, trial,
  detail`.
- `touch_data_*.csv` is the continuous slider recording. Columns: `elapsed_s,
  monotonic_s, position, touch_size, is_touching, timestamp_us`. `position`
  (0–1) is NaN whenever `is_touching == 0`.
- `distress_rating_*.csv/.xlsx` is **ignored**.

Both files carry `monotonic_s`, the clock shared by the event log and the
touch data. It is the only time reference used.

## Setup

Requires Python 3.10+ and the packages in [`requirements.txt`](requirements.txt).
Install with:

```bash
pip install -r requirements.txt
```

> Note: `lib/config.py` still imports `neurokit2`, which this branch does not
> otherwise use. Install it, or remove that import, for the pipeline to import.

## Running

```bash
python main.py          # batch over all JS_* folders in Data/
```

To process a single participant:

```python
from pathlib import Path
from lib.CAL_process import full_process_single

participant_id, df_vas, df_sound = full_process_single(Path("Data/JS_01"), show=False)
```

> `main.py` runs with `show=True`, which opens **two blocking windows per
> sound**: the trace plot and the histogram, so 276 windows per participant.
> Use `show=False` for batch runs. Plots are saved either way.

Output is written to `Results/`:

| File                        | Content                                       |
|-----------------------------|-----------------------------------------------|
| `extracted_VAS_results.csv` | Per-sound three-phase VAS metrics (long format) |

## Pipeline

`full_process_single(participant_path, show=False)` in
[`lib/CAL_process.py`](lib/CAL_process.py) runs these steps:

1. **Load touch data.** `load_touch_vas` (`lib/Metric_extraction/VAS_extract.py`)
   returns `monotonic_s`, `vas` (= `position × 100`, on a 0–100 scale) and
   `is_touching`.
2. **Verify and clean.** `load_and_clean_events`
   (`lib/PPG_extract/load_and_clean_events.py`) loads the event log, runs
   `verify_participant`, and trims events and touch data to the task block.
3. **Match sounds to touch data.** `match_sound_to_touch`
   (`lib/Metric_extraction/match_sound.py`) returns one row per task sound,
   holding the touch samples during the sound and in a ±3 s window around it.
4. **Per-sound loop.** For each sound it:
   - computes the metrics with `get_vas_phase_metrics`
     (`lib/Metric_extraction/metric_derive.py`), which also plots a histogram,
   - plots the trace with its phase markers (`lib/Plots_func/plot_sound.py`).

It returns `(participant_id, df_vas, df_sound)`.

### Verification

`verify_participant(events, participants_path)` returns two booleans:

- **Participant:** the `detail` of the `experiment_start` row holds
  `participant=###`. Its number must equal the trailing number of the folder
  name. A mismatch raises an error.
- **Trials:** `trial` must run 1–138 (`MAX_TASK_TRIALS`), with 138 distinct
  values. A mismatch only prints a warning.

### Cleaning

`load_and_clean_events` keeps the span from `block_start` to `block_end`.
Silences between sounds (`sound_gap_start`/`sound_gap_end`) and task pauses
(`task_pause_start`/`task_pause_end`), listed as `EXCLUDE_PAIRS` in
`lib/config.py`, are **kept** in both the events and the VAS data. They are only
recorded in `skips`, so the time the slider takes to return to 0 can be plotted.

`monotonic_s` is zero-referenced to the first kept event and rounded to the
microsecond. Time is **not** collapsed: events, touch samples and `skips` share
the same continuous clock.

It returns `(df_events, df_touch, skips)`. `skips` has one row per interval:
`type` (`sound_gap` or `task_pause`), `start`, `end` and `duration`.

### Sound matching

`match_sound_to_touch` uses `block == 'task'` events only, so the 14 training
sounds are excluded. Each sound is a `sound_play_start`/`sound_play_end` pair
with the same `trial`. Returned columns (`df_sound`):

| Column          | Description |
|-----------------|-------------|
| `trial`         | `1`–`138`. |
| `category`      | Sound category (the part of `detail` before `/`), e.g. `Bruits routiers`. |
| `sound_name`    | Sound file (the part after `/`), e.g. `Circulation_8.wav`. |
| `start_s`       | Sound start, on the zero-referenced `monotonic_s` clock. |
| `end_s`         | Sound end, on the same clock. |
| `duration_s`    | `end_s - start_s`. |
| `n_samples`     | Touch samples during the sound. |
| `touch`         | Touch samples in `[start_s, end_s)`, used for the metrics. |
| `touch_context` | Touch samples in `[start_s − 3 s, end_s + 3 s]` (`SOUND_PAD_S`), used for plotting. The windows of neighbouring sounds may overlap. |

Both `touch` and `touch_context` hold `monotonic_s`, `vas`, `is_touching` and
`t_rel_s` (time from sound onset).

## Output schema

The output is long format, with one row per `(sound × Metric)`:

| Column           | Description |
|------------------|-------------|
| `participant`    | Folder name, e.g. `JS_01`. |
| `trial`          | `1`–`138`. |
| `category`       | Sound category. |
| `sound_name`     | Sound file name. |
| `interval_start` | Sound start (`monotonic_s`, seconds). |
| `interval_end`   | Sound end (`monotonic_s`, seconds). |
| `task_moment`    | Always `task`. |
| `Metric`         | Metric name (see [Metrics](#metrics)). |
| `Value`          | The metric value. NaN when it cannot be computed. |
| `unit`           | Unit of `Value`: `VAS (0-100)` (slider position), `s` (duration), `s from sound onset` (time point), `VAS/s` (rate) or `count`. Set in `VAS_METRIC_UNITS` (`lib/config.py`). |
| `sample_size`    | Number of VAS samples during the sound. |
| `status`         | The sound's flags (see [Status flags](#status-flags)). The same value appears on every row of a sound. |
| `error`          | Why `Value` is missing for this metric, or a caveat about it. Empty when the value is valid. |

Every sound has the same 15 core rows. Each additional plateau adds 3 more rows
(`plateau_level_k`, `rise_dur_k`, `rise_rate_k`).

## Metrics

All metrics are computed per sound by `get_vas_phase_metrics`
(`lib/Metric_extraction/metric_derive.py`), over the samples during the sound.
The percentage thresholds follow the time-intensity trapezoid of Lallemand et
al. (1999). All parameters are in `lib/config.py`.

### Reference level

| `Metric` | Definition |
|----------|------------|
| `i_max`  | Highest VAS value after the first 2 s of the sound (`VAS_IMAX_EXCLUDE_S`). The first 2 s are excluded because the slider may still hold the previous sound's value. |

### Phase 1: delay

| `Metric`  | Definition |
|-----------|------------|
| `t_start` | Start of the rise (s from sound onset): the **last upward crossing of the onset threshold before `t_90`**. An upward crossing is a sample above the threshold whose previous sample was at or below it, so a value carried over from the previous sound or a brief blip is not taken as the start. |

**Onset threshold.** The threshold is `max(5 % of i_max, min(8, 50 % of i_max))`:

- `VAS_ONSET_FRAC = 0.05` is the Lallemand 5 % threshold.
- `VAS_REST_MAX = 8` is a resting tolerance. Many participants rest the slider at about 4–6 between sounds, not at 0, so any position up to 8 counts as rest.
- `VAS_REST_CAP_FRAC = 0.5` caps the tolerance at half of `i_max`, so that low ratings (e.g. a judgement of 8) still get a `t_start`.

The same threshold is the minimum level of a plateau.

### Phase 2: adjustment

| `Metric`          | Definition |
|-------------------|------------|
| `t_90`            | First time the VAS reaches 90 % of `i_max` (`VAS_RISE_FRAC`). |
| `rise_dur`        | `t_90 − t_start` (s). |
| `rise_rate`       | `(VAS(t_90) − VAS(t_start)) / rise_dur` (VAS points/s). NaN for a jump (see `JUMP`). |
| `n_plateaus`      | Number of distinct plateaus after `t_start`. A **plateau** is a run of consecutive samples in the same 1-point histogram bin lasting **≥ 2 s** (`VAS_PLATEAU_MIN_S`), with a level above 5 % of `i_max`. Neighbouring plateaus in the same bin are merged. |
| `n_changes`       | `n_plateaus − 1`: the number of moves from one plateau to the next. |
| `n_up`, `n_down`  | How many of those moves go up and how many go down. |
| `plateau_level_k` | Level of plateau *k*: the median VAS of its samples. |
| `rise_dur_k`      | Duration of the move into plateau *k*. It starts at `t_start` for *k* = 1, or at the end of plateau *k*−1, and ends when the VAS has covered 90 % of the step to plateau *k*. |
| `rise_rate_k`     | Signed rate of that move (negative for a drop). NaN for a jump. |

The overall rise (`t_90`, `rise_dur`, `rise_rate`) is measured against
`i_max`. The per-plateau rises (`*_k`) are measured against each plateau's own
level. With several plateaus the two can differ a lot. For example, with
plateaus at 61 and then 73, `rise_dur` is about 10.8 s, while `rise_dur_1` and
`rise_dur_2` are each under 1 s.

### Phase 3: plateau (final judgement)

| `Metric`        | Definition |
|-----------------|------------|
| `plateau_level` | The level the slider stayed at longest from `t_start` to the end of the sound, from a histogram with 1-point bins (`VAS_MODE_BIN_WIDTH`). Bins include their lower edge but not their upper edge, so 85.6 falls in bin [85, 86). The value reported is the **median of the samples inside the most populated bin**, i.e. the value actually held, not the bin centre. |
| `plateau_dur`   | Total time spent in plateaus (≥ 2 s) at the `plateau_level` bin. |
| `plateau_std`   | Standard deviation of the VAS from `t_90` to the end of the sound (`ddof=1`). |

### Zero judgements

When the VAS stays **below 5** (`VAS_NO_RESPONSE`) after the first 2 s, *and*
the finger is on the slider (`is_touching == 1`) for at least 80 % of the sound
(`VAS_MIN_TOUCH_FRAC`), the sound is a **valid judgement of 0**. This is common
for relaxing sounds. It is not an error:

- `plateau_level` is the median VAS after 2 s, so about 0.
- `plateau_dur` is the time spent below 5.
- `n_plateaus` is 1.
- Every other metric is **0**, and `error` is empty.

> `t_start`, `t_90`, `rise_dur` and `rise_rate` are 0 for these sounds. In a
> plain average that is indistinguishable from an instant response. To avoid
> this, exclude `status == "ZERO_JUDGEMENT"` from analyses of timing metrics.

### Status flags

`status` is `OK` or a `;`-separated list of flags.

The following flags describe why metrics are missing:

| Flag | Meaning | Metrics |
|------|---------|---------|
| `NO_DATA`        | No VAS samples during the sound. | All NaN |
| `NO_RESPONSE`    | The slider was never touched, or the VAS stayed below 5 with the finger on the slider less than 80 % of the time. | All NaN, except `i_max` when available |
| `TOO_SHORT`      | The sound is shorter than the 2 s excluded from `i_max`. | All NaN |
| `CARRY_OVER`     | The VAS never dropped below the onset threshold, so there is no `t_start`. Typically the slider stayed at the previous sound's rating. | All NaN except `i_max` |
| `ZERO_JUDGEMENT` | A valid judgement of 0 (see above). | Filled (0) |

The following flags can be combined:

| Flag | Meaning |
|------|---------|
| `NO_T90`        | The VAS never reached 90 % of `i_max`. `t_90`, `rise_dur`, `rise_rate` and `plateau_std` are NaN. |
| `NO_PLATEAU`    | No value was held for ≥ 2 s after `t_start`. `n_plateaus` = 0 and the plateau-1 metrics are NaN. |
| `MULTI_PLATEAU` | Two or more plateaus. |
| `SPIKE`         | `i_max` is more than 10 points above the highest plateau (`VAS_PEAK_MARGIN`), for **less than 0.3 s** (`VAS_SPIKE_MAX_S`): a brief artefact. `t_90` and the overall rise are then measured against the spike. |
| `OVERSHOOT`     | Same rule as `SPIKE`, but the excursion lasts **0.3 s or more**: the slider genuinely went past the judgement before settling. |
| `JUMP`          | A rise took less than 50 ms (`VAS_JUMP_MAX_S`), meaning the participant tapped a new position. The rate of that rise is NaN. |

Every missing `Value` has an explanation in `error`. For short sounds the
message gives the sound's timing, e.g. `no plateau possible: only 1.3 s between
t_start (2.6 s) and sound end (3.9 s), < 2 s minimum`.

## Plots

Both plots are saved by default with the formats in `SAVE_FORMATS`
(`lib/Plots_func/plot_config.py`). They are also shown when `show=True`.

- **Trace:** `plot_sound_touch` saves to `Plots/sound_touch/<category>/`.
  - It shows the VAS from 3 s before to 3 s after the sound, with time measured
    from sound onset.
  - Shading: the sound in blue, sound gaps in grey, task pauses in orange.
  - Phase markers: dotted lines at `t_start` and `t_90`, each plateau as a
    numbered red bar, and a ▼ where each per-plateau rise reaches 90 %.
- **Histogram:** `_plot_histogram_mode` saves to `Plots/sound_hist/<category>/`.
  It is called from inside `_histogram_mode`.
  - It shows the distribution of VAS values from `t_start` to the end of the
    sound.
  - Bar height is the **dwell time in seconds**, computed from each sample's
    real duration.
  - The `plateau_level` bin is highlighted in red.

## Limitations

The metrics are built around the **typical judgement pattern**: delay, then a
single adjustment, then a stable plateau. Treat them with caution outside that
pattern.

1. **More than two plateaus.** The code assigns plateau numbers and rises to
   any number of plateaus, but with more than two it struggles to tell the
   metrics apart:
   - The overall rise (`t_90`, `rise_dur`, `rise_rate`) is measured against
     `i_max`, which may belong to any plateau or excursion.
   - `plateau_level` is simply the level held longest, which is not
     necessarily the final judgement.
   - `plateau_std` mixes several levels.
   - Brief holds under 2 s between plateaus are not counted.

   Do not interpret sounds with `n_plateaus > 2` without inspecting their plot.
   Some of these sounds may be genuine re-judgements of changes within the
   sound.
2. **Very short sounds.** Some sounds last about 1.5–4 s (in JS_01: trial 116
   is 1.5 s, and trials 50, 63 and 102 are 3.75–3.9 s). The typical
   delay alone is about 2–3 s, so these sounds cannot produce reliable metrics:
   - Sounds under 2 s have no `i_max` (`TOO_SHORT`).
   - Sounds where less than 2 s remains after `t_start` cannot have a plateau
     (`NO_PLATEAU`).
   - Short sounds may not reach `t_90` (`NO_T90`).

   Values computed on short sounds, such as `plateau_level` without a held
   plateau, describe an unfinished judgement. The `error` column states the
   timing constraint.
3. **Atypical judgements.** When a judgement does not follow delay →
   adjustment → plateau, the same metrics are not entirely reliable. Examples:
   a slow continuous drift, repeated corrections, a judgement made before the
   2 s exclusion, a slider held over from the previous sound, or a tap instead
   of a slide.
   - `SPIKE` and `OVERSHOOT` mean the 5 %/90 % thresholds come from an
     excursion rather than the judgement.
   - `CARRY_OVER` and `JUMP` mean the timing or rate could not be measured.
   - A slider that wobbles across a 1-point bin edge can split a real hold into
     two bins, so the hold is missed as a plateau.
   - A judgement barely above the resting level (about 10 or less) is hard to
     separate from rest. The 8-point resting tolerance can treat it as rest, so
     `t_start` then points to a later movement. JS_10 trial 74 is an example.
   - Check the `status` flags and the plot before using these sounds.
4. **Duplicated timestamps.** About a third of the touch samples share their
   timestamp with the previous sample. Durations use each sample's real
   duration, so they are unaffected. However, an instantaneous rise between two
   such samples shows as "0 ms" (`JUMP`), and the histogram mode counts samples
   rather than time.
5. **Thresholds are choices.** The 5 %/90 % thresholds come from the
   literature (Lallemand et al., 1999). The others are our own choices, set
   from JS_01 and not validated across participants. They are all set in
   `lib/config.py`:
   - the 8-point resting tolerance and its 50 % cap,
   - the 2 s exclusion and 2 s plateau minimum,
   - 1-point bins,
   - the 5-point zero-judgement limit and 80 % touch fraction,
   - the 10-point and 0.3 s spike/overshoot rules,
   - the 50 ms jump limit.
