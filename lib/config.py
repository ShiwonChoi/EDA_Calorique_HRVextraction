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


# paired markers whose enclosed events/samples are removed
EXCLUDE_PAIRS = [
    ("sound_gap_start", "sound_gap_end"),
    ("task_pause_start", "task_pause_end"),
]

# ============================================================================
# VAS PARAMETERS
# ============================================================================
MAX_TASK_TRIALS = 138


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
    'Metric', 'Value',
    'sample_size', 'status', 'error',
]

# Bin width (VAS 0-100 scale) used to compute the histogram mode of the VAS
VAS_MODE_BIN_WIDTH = 1.0
