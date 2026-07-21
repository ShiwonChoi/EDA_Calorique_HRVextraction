import pandas as pd
from pathlib import Path
import datetime as datetime
import re
import neurokit2 as nk


# Conditions and groupes
############################################################################

ALLCOND = [
    #('shimmer_connected', 'Shimmer Connected'),
    #('experiment_start', 'Experiment Start'),
    #('instruction_start', 'Instruction Start'),
    #('instruction_end', 'Instruction End'),
    ('baseline_start', 'Baseline Start'),
    ('baseline_end', 'Baseline End'),
    ('block_start', 'Block Start'),
    ('countdown_start', 'Countdown Start'),
    #('countdown_end', 'Countdown End'),
    ('sound_play_start', 'Sound Play Start'),
    ('sound_play_end', 'Sound Play End'),
    ('block_end', 'Block End'),
    ('rest_start', 'Rest Start'),
    ('rest_end', 'Rest End'),
    ('post_recovery_start', 'Post Recovery Start'),
    ('post_recovery_end', 'Post Recovery End'),
    #('thankyou_screen', 'Thank You Screen'),
    #('experiment_end', 'Experiment End'),
]


# Default frequency grid — log-spaced for finer resolution in LF/HF
DEFAULT_FREQ_MIN = 0.02
DEFAULT_FREQ_MAX = 0.50
DEFAULT_N_FREQS  = 100

# Adaptive Morlet n_cycles ramp
DEFAULT_N_CYCLES_LOW  = 5
DEFAULT_N_CYCLES_HIGH = 15

# Default reflection-padding length (seconds per side).
# Sized to suppress COI at DEFAULT_FREQ_MIN:
#   max_half_width = 3 * N_CYCLES_LOW / (2π * FREQ_MIN) ≈ 120s for 0.02 Hz
DEFAULT_PAD_SECONDS = 120


# ============================================================================
# HRV FREQUENCY BANDS (Task Force 1996) — frequency-domain only
# ============================================================================
HRV_BANDS = {
    'VLF': (0.003, 0.04),
    'LF':  (0.04,  0.15),
    'HF':  (0.15,  0.40),
}

BAND_COLORS = {
    'VLF':         '#ff9999',
    'LF':          '#99ccff',
    'HF':          '#99ff99',
    'LF_HF_ratio': 'black',
}


# ============================================================================
# CWT (CONTINUOUS WAVELET TRANSFORM) PARAMETERS
# ============================================================================
# Frequency grid: log-spaced gives finer resolution where the
# physiology lives (LF and HF bands)
CWT_FREQ_MIN = 0.01
CWT_FREQ_MAX = 0.50
CWT_N_FREQS  = 100

# Adaptive Morlet n_cycles ramp:
#   - high n_cycles at low freq -> long wavelet -> fine freq resolution
#   - low  n_cycles at high freq -> short wavelet -> fine time resolution
CWT_N_CYCLES_LOW  = 5
CWT_N_CYCLES_HIGH = 15


# ============================================================================
# GSR / EDA (ELECTRODERMAL ACTIVITY) PARAMETERS
# ============================================================================
# Physiologically-plausible conductance range and max slope, used to flag
# artifacts (e.g. Shimmer auto-range switching) before tonic/phasic
# decomposition. Source: Kleckner et al. (2018), IEEE TBME 65(7), 1460-1467.
GSR_PLAUSIBLE_RANGE_US  = (0.05, 60.0)   # microSiemens
GSR_MAX_SLOPE_US_PER_S  = 10.0           # microSiemens / second

# Tonic/phasic decomposition method, passed through to nk.eda_phasic.
# 'highpass' (default, no extra deps) and 'cvxeda' (Greco et al. 2016, needs
# the cvxopt package) are the two primary supported options; 'sparse'
# (Hernando-Gallego et al. 2018) is available but flagged experimental by
# NeuroKit2 itself.
GSR_DECOMPOSITION_METHOD = 'highpass'

# Minimum SCR amplitude (microSiemens) for nk.eda_peaks to count a phasic
# peak as a genuine skin conductance response. 0.1 matches NeuroKit2's own
# default; the EDA literature also uses 0.05 (legacy) or 0.01 (modern
# high-resolution digital systems) — adjust here if needed.
SCR_AMPLITUDE_MIN_US = 0.1


# Pathways
############################################################################

# Get the directory where the script itself is located
main_dir = Path(__file__).parent

# Get the project root (one level above the lib folder)
base_dir = Path(__file__).resolve().parents[1]

# Build paths to Data folders
DATA_DIR = base_dir / "Data"
CORR_DIR = DATA_DIR / "Processed_PPG"

# Participant folders: root of raw per-participant data (each SBSA_##/SBA_## lives here)
PARTICIPANTS_DIR = DATA_DIR

# Output folder for processed PPG results
PROCESSED_PPG_DIR = DATA_DIR / "Processed_PPG"
PROCESSED_PPG_DIR.mkdir(parents=True, exist_ok=True)

# Directory for results
output_dir = base_dir / "Results"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "processed_ppg_results.csv"


def get_participant_paths(root=PARTICIPANTS_DIR, patterns=("SBSA_*", "SBA_*")):
    """Return sorted list of participant folders found under root.

    The sound-stress cohort uses two folder prefixes:
        SBSA_## — healthy controls
        SBA_## — tinnitus (acouphène) group
    Both are collected. `patterns` may be a single glob string or an iterable
    of glob strings.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    paths = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    return sorted(paths)


# Study-group codes keyed by folder prefix.
GROUP_MAP = {
    "SBSA": "HC",   # healthy controls
    "SBA": "T",    # tinnitus (acouphène)
}


def group_from_participant(participant_id):
    """Map a participant/folder id to its study-group code.

    SBSA_* -> 'HC' (healthy controls); SBA_* -> 'T' (tinnitus / acouphène).
    Returns pd.NA for anything that matches neither prefix.
    """
    pid = str(participant_id).upper()
    for prefix, code in GROUP_MAP.items():
        if pid.startswith(prefix):
            return code
    return pd.NA


# ============================================================================
# UNIFIED OUTPUT SCHEMA — shared by CAL_temp and CAL_freq (total + interval)
# ============================================================================
OUTPUT_COLUMNS = [
    'participant', 'trial', 'condition',
    'time_interval_rel_start', 'time_interval_abs_start',
    'time_interval_rel_end',   'time_interval_abs_end',
    'task_moment', 'recording_type',
    'Metric', 'Value_type', 'Value',
    'sample_size', 'status', 'error',
]

# Shared across temporal and frequency HRV metrics.
# Baseline rows carry raw values; diff/pct_change/log_ratio are set to 0.0
#temporal: by _derive_baseline_corrected; frequency: filled in binning functions
# since per-frequency correction is not run against a trial's own data).
VALUE_TYPES = ['raw', 'diff', 'pct_change', 'log_ratio']