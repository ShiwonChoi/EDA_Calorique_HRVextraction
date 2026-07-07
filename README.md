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
turns the raw per-trial recordings into clean, baseline-referenced HRV (and,
on some branches, EDA) feature tables ready for statistical analysis.

**Author:** Shiwon Choi

---

## Repository branches

The project spans three branches, each serving a distinct purpose. They share
the same core architecture (`lib/`, `main.py`, unified output schema) but
target different experiments or serve as an archive.

| Branch | Status | Purpose |
|--------|--------|---------|
| **`main`** | **Active** | Canonical **PPG → HRV** pipeline for the caloric-stress (SC) cohort. Extracts temporal and frequency-domain HRV per trial, baseline-referenced against each participant's own baseline. This is the branch documented in detail below. |
| **`SoundStress_HRV`** | **Active** | The **Sound Stress project** — a *separate analysis* of the **SBSA** participant files. Adapts the same PPG/HRV extraction to the Sound Stress experiment, whose acquisition format differs from the SC study (session-based `shimmer`, `event_log`, `distress_rating`, and `touch_data` files rather than the SC per-trial layout). This is the branch used for the SoundStress analysis. |
| **`Nested_HRVextraction_archive`** | **Archived — not used** | A frozen snapshot kept "just in case." Preserves the **old nested folder architecture** (everything under a duplicated `EDA_calorique_HRVextraction/` subfolder) together with the early **GSR / electrodermal-activity (EDA) planning documents** — `GSR_EDA_extraction_plan.md` and a peer-review-grounded `GSR_EDA_literature_review.md`. These are *planning and literature only*; no EDA extraction code was implemented. Not maintained; retained purely for reference. |

> **Note on the naming.** These branches were previously named
> `EDA_SoundStress_HRVextraction` (now `Nested_HRVextraction_archive`) and
> `EDA_SoundStress_HRVextraction_v2` (now `SoundStress_HRV`). They were renamed
> to make the versions and their intent unambiguous: `SoundStress_HRV` is the
> live SoundStress/SBSA analysis, and `Nested_HRVextraction_archive` is a
> read-only archive of the old nested layout and EDA planning notes.

---

## What the pipeline does (`main` branch)

`main.py` batch-processes every `SC_*` participant folder and produces two
consolidated HRV tables. For each participant, `full_process_single`
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
6. **Baseline referencing** — every stimulation metric is expressed relative to
   the participant's own baseline (`Trial00`) as `raw`, `diff`, `pct_change`,
   and `log_ratio` value types.

Both temporal and frequency outputs share a single unified schema
(`OUTPUT_COLUMNS` in `lib/config.py`), so the two tables are directly stackable.

---

## Requirements

- **Python 3.9+**
- Python packages:
  - `neurokit2`
  - `pandas`
  - `numpy`
  - `scipy`
  - `matplotlib`

Install with:

```bash
pip install neurokit2 pandas numpy scipy matplotlib
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
- The Shimmer CSV carries both the PPG channel (used here) and the GSR channel
  (consumed by the EDA branch).

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

Two CSV files are written to `Results/` (folder is git-ignored):

| File | Contents |
|------|----------|
| `processed_ppg_results_temp.csv` | Temporal HRV metrics (`mean_HR`, `mean_RRI`, `RMSSD`, `SDNN`) — whole-trial and 30-s bins. |
| `processed_ppg_results_freq.csv` | Frequency-domain band power (`VLF`, `LF`, `HF`) — whole-trial and 30-s bins. |

Both files use the same long-format schema (`OUTPUT_COLUMNS`):

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
- `sample_size` reports `"<n_clean> / <n_raw>"` beats after artifact removal.

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
└── Metric_extraction/
    ├── RRI_preprocess.py       # Artifact removal, resampling, detrending
    ├── HRV_temp_extract.py     # Temporal HRV metrics
    ├── HRV_temp_bin.py         # 30-s temporal binning
    ├── HRV_freq_extract.py     # CWT band-power extraction
    ├── HRV_freq_bin.py         # Total / 30-s frequency binning
    └── HRV_df.py               # Unified output-row builder
```

---

## Author

**Shiwon Choi**

## License

See [LICENSE](LICENSE).
