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


# Pathways
############################################################################

# Get the directory where the script itself is located
main_dir = Path(__file__).parent

# Get the project root (one level above the lib folder)
base_dir = Path(__file__).resolve().parents[1]

# Build paths to Data folders
DATA_DIR = base_dir / "Data" 
CORR_DIR = base_dir / "Data" / "Processed_PPG"

# Directory for results
output_dir = base_dir / "Results"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "processed_ppg_results.csv"