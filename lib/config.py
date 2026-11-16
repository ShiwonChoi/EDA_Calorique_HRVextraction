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


# paired markers recorded as skips (silence between sounds / task pauses).
# They are kept in the events and VAS data, only logged in `skips`.
EXCLUDE_PAIRS = [
    ("sound_gap_start", "sound_gap_end"),
    ("task_pause_start", "task_pause_end"),
]

# ============================================================================
# VAS PARAMETERS
# ============================================================================
MAX_TASK_TRIALS = 138

# seconds of VAS data shown before and after each sound in the sound plots
SOUND_PAD_S = 3.0


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
output_file = output_dir / "extracted_VAS_results.csv"


def get_participant_paths(root=PARTICIPANTS_DIR, patterns=("JS_*",)):
    """Return sorted list of participant folders found under root.

    The sound-stress cohort uses two folder prefixes:
        JS_## — healthy controls
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
    "JS": "JS",   # healthy controls
}




# ============================================================================
# OUTPUT SCHEMA — one row per sound per metric
# ============================================================================
OUTPUT_COLUMNS = [
    'participant', 'trial', 'category', 'sound_name',
    'interval_start', 'interval_end',
    'task_moment',
    'Metric', 'Value', 'unit',
    'sample_size', 'status', 'error',
]

# Unit of each VAS metric (written to the 'unit' column of the output).
# Per-plateau metrics (<name>_1, <name>_2, ...) take the unit of <name>.
VAS_METRIC_UNITS = {
    'i_max':         'VAS (0-100)',
    't_start':       's from sound onset',
    't_90':          's from sound onset',
    'rise_dur':      's',
    'rise_rate':     'VAS/s',
    'n_plateaus':    'count',
    'n_changes':     'count',
    'n_up':          'count',
    'n_down':        'count',
    'plateau_level': 'VAS (0-100)',
    'plateau_dur':   's',
    'plateau_std':   'VAS (0-100)',
}

# Bin width (VAS 0-100 scale) used to compute the histogram mode of the VAS
VAS_MODE_BIN_WIDTH = 1.0

# ---- three-phase VAS metrics (latency -> adjustment -> judgement) ----------
# Thresholds follow the time-intensity trapezoid (Lallemand et al., 1999)
VAS_IMAX_EXCLUDE_S = 2.0   # i_max ignores the first N s (carry-over from previous sound)
VAS_ONSET_FRAC     = 0.05  # t_start: upward crossing of 5 % of i_max ...
VAS_REST_MAX       = 10.0   #   ... but at least 10 (resting tolerance: slider often rests at ~4-6) ...
VAS_REST_CAP_FRAC  = 0.50  #   ... capped at 50 % of i_max so low ratings keep a t_start
VAS_RISE_FRAC      = 0.90  # t_90: first reach of 90 % of i_max (also 90 % of each plateau step)
VAS_PLATEAU_MIN_S  = 2.0   # plateau = same histogram bin for >= N s consecutively
VAS_NO_RESPONSE    = 5.0   # i_max below this (VAS points) = near-0 sound:
VAS_MIN_TOUCH_FRAC = 0.80  #   finger on slider >= this fraction -> ZERO_JUDGEMENT, else NO_RESPONSE
VAS_PEAK_MARGIN    = 10.0  # i_max > highest plateau + N points -> peak excursion, then:
VAS_SPIKE_MAX_S    = 0.30  #   excursion shorter than N s -> SPIKE, otherwise -> OVERSHOOT
VAS_JUMP_MAX_S     = 0.05  # rise faster than N s = tap/jump -> rise_rate NaN, JUMP flag
