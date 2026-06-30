import pandas as pd
from pathlib import Path
import datetime as datetime
import re
import neurokit2 as nk


# Conditions and groupes
############################################################################

ALLBASECOND = [
    #('shimmer_connected', 'Shimmer Connected'),
    #('baseline_recording_start', 'Baseline Recording Start'),
    ('baseline_start', 'Baseline Start'),
    ('baseline_end', 'Baseline End'),
    #('baseline_recording_stop', 'Baseline Recording Stop')
]

ALLSTIMCOND = [
    #('shimmer_connected', 'Time Start'),
    #('trial_recording_start', 'Recording Start'),
    #('trial_start', 'Trial Start'),
    ('pre_stimulation_baseline_start', 'Anticipation Start'),
    ('pre_stimulation_baseline_end', 'Anticipation End'),
    #('vibration_start', 'Vibration Start'),
    #('countdown_start', 'Countdown Start'),
    #('countdown_end', 'Countdown End'),
    ('stimulation_start', 'Task Start'),
    #('vibration_stop', 'Vibration Start'),
    #('vibration_count_final', 'Vibration Count End'),
    ('recovery_start', 'Recovery Start'),
    ('stimulation_end', 'Stimulation End'),
    ('recovery_end', 'Recovery End'),
    #('trial_end', 'Trial End'), 
    #('trial_recording_stop', 'Trial Recording End')
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



# Pathways
############################################################################

# Get the directory where the script itself is located
main_dir = Path(__file__).parent

# Get the project root (one level above the lib folder)
base_dir = Path(__file__).resolve().parents[1]

# Build paths to Data folders
DATA_DIR = base_dir / "Data"
CORR_DIR = DATA_DIR / "Processed_PPG"

# Participant folders: root of raw per-participant data (each SC_## lives here)
PARTICIPANTS_DIR = DATA_DIR

# Output folder for processed PPG results
PROCESSED_PPG_DIR = DATA_DIR / "Processed_PPG"
PROCESSED_PPG_DIR.mkdir(parents=True, exist_ok=True)

# Directory for results
output_dir = base_dir / "Results"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "processed_ppg_results.csv"


def get_participant_paths(root=PARTICIPANTS_DIR, pattern="SC_*"):
    """Return sorted list of SC_## participant folders found under root."""
    return sorted(root.glob(pattern))