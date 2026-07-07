# EDA_calorique_HRVextraction

Physiological-signal extraction and analysis for the **Stress Caloric ("SC")
study** — a caloric-vestibular stimulation experiment that pairs autonomic
measures (heart-rate variability from PPG, electrodermal activity from GSR)
with **caloric nystagmus** recordings, in order to study how vestibular stress
manifests in the autonomic nervous system.

Each participant undergoes a baseline recording followed by caloric-stimulation
trials. During those trials the Shimmer wearable records photoplethysmography
(PPG) and galvanic skin response (GSR), while eye-tracking captures the
nystagmus (and VOR) response evoked by the caloric stimulus. This repository
turns the raw per-trial recordings into clean, baseline-referenced HRV and
EDA (electrodermal activity / GSR) feature tables ready for statistical
analysis.

**Author:** Shiwon Choi

---

## Repository branches

The project spans three branches, each serving a distinct purpose. They share
the same core architecture (`lib/`, `main.py`, unified output schema) but
target different experiments or serve as an archive.

| Branch | Status | Purpose |
|--------|--------|---------|
| **`main`** | **Active** | Canonical **PPG → HRV** and **GSR → EDA** pipeline for the caloric-stress (SC) cohort. Extracts temporal and frequency-domain HRV, plus tonic/phasic EDA, per trial, baseline-referenced against each participant's own baseline. This is the branch documented in detail below. |
| **`SoundStress_HRV`** | **Active** | The **Sound Stress project** — a *separate analysis* of the **SBSA/SBAA** participant files. Adapts the same PPG/HRV/EDA extraction (plus a subjective-stress VAS trace) to the Sound Stress experiment, whose acquisition format differs from the SC study (one continuous session-based `shimmer`/`event_log`/`touch_data` recording rather than the SC per-trial layout). This is the branch used for the SoundStress analysis. |
| **`Nested_HRVextraction_archive`** | **Archived — not used** | A frozen snapshot kept "just in case." Preserves the **old nested folder architecture** (everything under a duplicated `EDA_calorique_HRVextraction/` subfolder) together with the early **GSR / electrodermal-activity (EDA) planning documents** — `GSR_EDA_extraction_plan.md` and a peer-review-grounded `GSR_EDA_literature_review.md`, which informed the EDA implementation later carried out on `main` and `SoundStress_HRV`. Not maintained; retained purely for reference. |

> **Note on the naming.** These branches were previously named
> `EDA_SoundStress_HRVextraction` (now `Nested_HRVextraction_archive`) and
> `EDA_SoundStress_HRVextraction_v2` (now `SoundStress_HRV`). They were renamed
> to make the versions and their intent unambiguous: `SoundStress_HRV` is the
> live SoundStress/SBSA analysis, and `Nested_HRVextraction_archive` is a
> read-only archive of the old nested layout and EDA planning notes.

---

## What the pipeline does (`main` branch)

`main.py` batch-processes every `SC_*` participant folder and produces three
consolidated tables (temporal HRV, frequency HRV, GSR/EDA). For each
participant, `full_process_single`
(`lib/CAL_process.py`) runs an independent pipeline **per trial**
(`Trial00` = baseline, followed by the caloric-stimulation trials). The
stimulation trials correspond to the four bithermal caloric irrigations:
**RW** (right warm), **LW** (left warm), **RC** (right cold), and **LC**
(left cold):

1. **Load & clean PPG** (`lib/PPG_extract/`) — read the per-trial
   `shimmer_*.csv`, attach event markers, and resample onto a uniform time axis.
2. **Peak detection & correction** — NeuroKit2 systolic-peak detection with
   support for manually corrected peaks, then convert to R-R intervals (RRI).
3. **RRI preprocessing** (`lib/Metric_extraction/RRI_preprocess.py`) —
   physiological/statistical artifact detection and removal, cubic-spline
   resampling to a uniform rate, and high-pass detrending for the
   frequency-domain analysis. Each stage is quality-validated and can be
   visualized (tachograms, preprocessing steps).
4. **Temporal HRV** (`HRV_temp_extract.py`) — `mean_HR`, `mean_RRI`, `RMSSD`,
   `SDNN` over the whole trial, plus optional **30-second bins** for
   stimulation trials.
5. **Frequency HRV** (`HRV_freq_extract.py`) — continuous wavelet transform
   (adaptive Morlet) to estimate band power in the **VLF / LF / HF** bands
   (Task Force 1996), with whole-trial and 30-second binned values.
6. **GSR/EDA** (`lib/GSR_extract/gsr_preprocess.py`) — the same trial's raw
   Shimmer `GSR` channel (skin resistance, kOhm) is converted to conductance
   (µS), artifact-checked, and split into a slow **tonic** (skin conductance
   level) and fast **phasic** (skin conductance response) component via
   NeuroKit2 (`gsr_method`, default `'highpass'`). `Tonic_SCL_mean/slope` and
   `Phasic_SCR_count/rate/amplitude/AUC` are computed whole-trial and in
   30-second bins, same as the HRV metrics.
7. **Baseline referencing** — every stimulation metric (HRV and GSR/EDA alike)
   is expressed relative to the participant's own baseline (`Trial00`) as
   `raw`, `diff`, `pct_change`, and `log_ratio` value types.

Temporal HRV, frequency HRV, and GSR/EDA outputs all share a single unified
schema (`OUTPUT_COLUMNS` in `lib/config.py`), so the three tables are directly
stackable.

---

## Requirements

- **Python 3.9+**
- Python packages (see `requirements.txt`):
  - `neurokit2` — peak detection, signal processing, EDA decomposition
  - `pandas`, `numpy`, `scipy`, `matplotlib`
  - `cvxopt` — only exercised if GSR/EDA is run with `gsr_method='cvxeda'`;
    the default `'highpass'` method doesn't need it

Install with:

```bash
pip install -r requirements.txt
```

---

## Input data

Research data is **not** stored in the repository (see `.gitignore`) — it lives
under a local `Data/` directory. Point the pipeline at it via `DATA_DIR` in
`lib/config.py` (default: `<repo>/Data`).

Expected per-participant layout (`main` / SC study):

```
Data/
├── SC_01/
│   ├── SC_01_VOR_initiale.csv          # VOR / nystagmus (baseline)
│   ├── SC_01_VOR_RW.csv                # VOR / nystagmus (right warm caloric)
│   ├── SC_01_VOR_LW.csv                # VOR / nystagmus (left warm caloric)
│   └── Stress measures/
│       ├── shimmer_P001_Trial00_baseline_<timestamp>.csv   # PPG + GSR
│       ├── shimmer_P001_Trial0*_RW_<timestamp>.csv         # right warm
│       ├── shimmer_P001_Trial0*_LW_<timestamp>.csv         # left warm
│       ├── shimmer_P001_Trial0*_RC_<timestamp>.csv         # right cold
│       ├── shimmer_P001_Trial0*_LC_<timestamp>.csv         # left cold
│       ├── events_P001_Trial0*_*.csv / .json               # event markers
│       ├── nystagmus_P001_Trial0*_*.csv                    # caloric nystagmus
│       └── state_001.json
├── SC_02/
└── ...
```

- Trials are discovered automatically from the `shimmer_*.csv` filenames.
- `Trial00` (`baseline`) is required per participant — it is the reference all
  stimulation trials are corrected against.
- The Shimmer CSV carries both the PPG channel (HRV) and the GSR channel
  (tonic/phasic EDA) — both are extracted by this branch.

---

## Usage

From the repository root:

```bash
python main.py
```

This will:

1. Discover all `SC_*` folders under `DATA_DIR`.
2. Process each participant/trial (progress and per-trial quality reports are
   printed to the console).
3. Concatenate results across the cohort and write them to `Results/`.

To process a single participant or enable diagnostic plots, call
`full_process_single(participant_path, show=True)` directly (see `main.py` and
`lib/CAL_process.py`). The `bin` argument controls the interval width in
seconds (default `30`).

---

## Output

Three CSV files are written to `Results/` (folder is git-ignored):

| File | Contents |
|------|----------|
| `processed_ppg_results_temp.csv` | Temporal HRV metrics (`mean_HR`, `mean_RRI`, `RMSSD`, `SDNN`) — whole-trial and 30-s bins. |
| `processed_ppg_results_freq.csv` | Frequency-domain band power (`VLF`, `LF`, `HF`) — whole-trial and 30-s bins. |
| `processed_ppg_results_gsr.csv`  | Tonic/phasic EDA metrics — whole-trial and 30-s bins. |

**HRV metrics** reflect autonomic activity on the heart (a mix of sympathetic
and parasympathetic/vagal influence): `mean_HR`/`mean_RRI` are average heart
rate and beat interval; `RMSSD` (beat-to-beat variability) is the standard
proxy for vagal/parasympathetic tone; `SDNN` is overall variability. `VLF`/
`LF`/`HF` are the same beat-interval series decomposed by oscillation speed —
`HF` tracks respiration-linked vagal activity, `LF` is traditionally linked to
sympathetic/baroreflex activity but is now considered a mixed signal.

**GSR/EDA metrics** reflect sweat-gland activity driven by the sympathetic
nervous system only (no vagal contribution), split into a slow **tonic**
background level and fast **phasic** response spikes:

| `Metric` | Component | Meaning |
|----------|-----------|---------|
| `Tonic_SCL_mean`            | Tonic  | Average background skin conductance (µS) — higher means more sustained sympathetic arousal. |
| `Tonic_SCL_slope`           | Tonic  | Linear trend of the background level (µS/s) — rising = building arousal, falling = habituation/recovery. |
| `Phasic_SCR_count`          | Phasic | Number of distinct skin-conductance-response spikes detected in the window. |
| `Phasic_SCR_rate`           | Phasic | Same count, normalized to responses per minute. |
| `Phasic_SCR_amplitude_mean` | Phasic | Average size (µS) of the detected responses. |
| `Phasic_SCR_amplitude_sum`  | Phasic | Total size (µS) of all detected responses added together. |
| `Phasic_AUC`                | Phasic | Area under the phasic curve (µS·s) — combines response count and size into one measure. |

All three files use the same long-format schema (`OUTPUT_COLUMNS`):

```
participant, trial, condition,
time_interval_rel_start, time_interval_abs_start,
time_interval_rel_end,   time_interval_abs_end,
task_moment, recording_type,        # recording_type: 'total' or 'interval'
Metric, Value_type, Value,          # Value_type: raw | diff | pct_change | log_ratio
sample_size, status, error
```

- `recording_type = 'total'` rows give the whole-trial metric; `'interval'`
  rows give the 30-second binned metric (stimulation trials only).
- `Value_type` encodes the baseline reference: baseline rows carry `raw`
  values (with `diff`/`pct_change`/`log_ratio` = 0.0 by convention), while
  stimulation rows carry all four value types relative to `Trial00`.
- `sample_size` reports `"<n_clean> / <n_raw>"` — beats after artifact removal
  for HRV rows, or non-artifact/total GSR samples in the window for EDA rows.

---

## Project layout

```
main.py                         # Batch entry point
lib/
├── config.py                   # Paths, HRV bands, CWT params, output schema
├── utils.py
├── CAL_process.py              # full_process_single — per-trial pipeline
├── PPG_extract/
│   ├── load_and_clean_ppg.py   # Load Shimmer CSV, attach events, resample
│   ├── ppg_preprocess.py       # PPG cleaning / peak detection
│   └── manual_peak.py          # Corrected-peak loading, RRI intervals
├── GSR_extract/
│   └── gsr_preprocess.py       # Conductance conversion, artifacts, tonic/phasic decomposition, SCR peaks
└── Metric_extraction/
    ├── RRI_preprocess.py       # Artifact removal, resampling, detrending
    ├── HRV_temp_extract.py     # Temporal HRV metrics
    ├── HRV_temp_bin.py         # 30-s temporal binning
    ├── HRV_freq_extract.py     # CWT band-power extraction
    ├── HRV_freq_bin.py         # Total / 30-s frequency binning
    ├── EDA_temp_extract.py     # Tonic/phasic EDA metrics
    ├── EDA_bin.py              # 30-s EDA binning
    └── HRV_df.py               # Unified output-row builder
```

---

## Author

**Shiwon Choi**

## License

See [LICENSE](LICENSE).
