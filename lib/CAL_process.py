import numpy as np
import pandas as pd
from pathlib import Path
from lib.config import OUTPUT_COLUMNS
from lib.PPG_extract.load_and_clean_events import load_and_clean_events
from lib.Metric_extraction.VAS_extract import load_touch_vas
from lib.Metric_extraction.metric_derive import get_vas_metrics
from lib.Metric_extraction.match_sound import match_sound_to_touch
from lib.Plots_func.plot_sound import plot_sound_touch


def full_process_single(participant_path, show=False):
    """
    Process one participant's continuous VAS recording.

    The event log and touch data are loaded, verified (participant id, 138
    trials) and cleaned once via load_and_clean_events, then each of the 138
    task sounds is matched to its touch data via match_sound_to_touch.

    Returns:
        participant_id (str)         : Participant identifier (e.g. "JS_##").
        df_vas         (pd.DataFrame): VAS rows, schema = OUTPUT_COLUMNS.
        df_sound       (pd.DataFrame): One row per sound (trial, category,
                                       sound_name, start_s, end_s, duration_s,
                                       n_samples, touch).
    """
    participant_path = Path(participant_path)
    participant_id   = participant_path.name
    print(f"\nParticipant folder : {participant_id}")

    CAL_vas  = []
    df_sound = pd.DataFrame()

    try:
        # 1. Grab touch data (events are loaded inside load_and_clean_events)
        df_touch = load_touch_vas(participant_path)
        if df_touch is None:
            print("  No VAS/touch data — VAS extraction skipped for this participant")
            return participant_id, pd.DataFrame(columns=OUTPUT_COLUMNS), df_sound

        # 2. Verify participant, trim events to the task, clean touch data around events
        df_events, df_touch, skips = load_and_clean_events(df_touch, participant_path, show=show)

        # 3. Associate each sound with its touch data
        df_sound = match_sound_to_touch(df_events, df_touch)
        print(f"  Matched {len(df_sound)} sounds to touch data")

        # 4. Per-sound loop: mean / mode / median / std of the VAS over the sound interval
        for row in df_sound.itertuples():
            try:
                # raw position over the sound interval
                plot_sound_touch(row.touch, row.trial, row.category, row.sound_name,
                                 row.start_s, row.end_s, show=show)

                # touch samples only while the slider is touched (position not NaN)
                touch = row.touch[row.touch["is_touching"] == 1]

                metrics, n = get_vas_metrics(touch["vas"].to_numpy(float))
                base = {
                    'participant': participant_id, 'trial': row.trial,
                    'category': row.category, 'sound_name': row.sound_name,
                    'interval_start': row.start_s, 'interval_end': row.end_s,
                    'task_moment': 'task',
                }
                if not metrics:
                    CAL_vas.append({**base, 'Metric': np.nan, 'Value': np.nan,
                                    'sample_size': 0, 'status': 'NO_DATA', 'error': ''})
                for metric_name, value in metrics.items():
                    CAL_vas.append({**base, 'Metric': metric_name, 'Value': float(value),
                                    'sample_size': n, 'status': 'OK', 'error': ''})

            except Exception as sound_err:
                import traceback
                print(f"\n  ERROR processing trial {row.trial} ({row.sound_name}): {sound_err}")
                traceback.print_exc()
                CAL_vas.append({
                    'participant': participant_id, 'trial': row.trial,
                    'category': row.category, 'sound_name': row.sound_name,
                    'interval_start': row.start_s, 'interval_end': row.end_s,
                    'task_moment': 'task', 'Metric': np.nan, 'Value': np.nan,
                    'sample_size': 0,
                    'status': 'FAILED', 'error': str(sound_err),
                })

        df_vas = pd.DataFrame(CAL_vas, columns=OUTPUT_COLUMNS) if CAL_vas else pd.DataFrame(columns=OUTPUT_COLUMNS)

        print(f"\n{'=' * 55}")
        print(f"  All sounds processed for {participant_id}.")
        print(f"  CAL_vas  rows : {len(df_vas)}")
        print(f"{'=' * 55}")

        return participant_id, df_vas, df_sound

    except Exception as e:
        import traceback
        print(f"\nERROR processing {participant_path.name}: {e}")
        traceback.print_exc()
        df_vas = pd.DataFrame(CAL_vas, columns=OUTPUT_COLUMNS) if CAL_vas else pd.DataFrame(columns=OUTPUT_COLUMNS)
        return participant_id, df_vas, df_sound
