import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import datetime as datetime
import matplotlib.pyplot as plt
import time
from lib.config import *
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg


#########################################################
#Implement process_single func 
#########################################################
def full_process_single(xdf_file):
    """
    Process one participant's data with event cleaning and segmentation.
    """

    try:
        # Load and preprocess PPG and EVENTS from xdf, raw PPG and raw EVENTs
        df_ppg, df_events, fs_ppg, badsegments_ppg = load_and_clean_ppg(xdf_file, show=False)

        # Replace / compare the automatic peaks to corrected peaks
        #-------------------------------------------------------------
        print(f"Adding manual peaks of {participant}...")
        signal, info = process_signal_with_peaks(
            df_ppg=df_ppg,
            fs_ppg=fs_ppg,
            xdf_file=xdf_file,
            participant_id=participant,
            pattern_hint = pattern_hint,
            corrected_dir=CORR_DIR  # or None to skip
        )


        ###############################
        # Clean events exercise 2_6
        ###############################
        print (f"Cleaning and trimming EVENT files and signal of {participant}...")

        # Modify df_events to follow structure: 
        # timestamps, Time (s), value -> timestamps, time_seconds, event
        df_events.rename(columns={
            'timestamps': 'timestamps',
            'Time (s)': 'time_seconds',
            'value': 'event'
            }, inplace=True)
        
        df_events.reset_index(drop=True, inplace=True) 


        # Extract intervals: baseline, anticipation, task, recovery
        # ----------------------------------------------------------------------------------------------------------

        print("\nExtracting NOISE intervals...")
        noise_intervals = extract_task_intervals('noise', signal_clean, df_events_clean, is_noiseonly=is_noiseonly)

        print("\nExtracting ARITH intervals...")
        arith_intervals = extract_task_intervals('arith', signal_clean, df_events_clean, is_noiseonly=is_noiseonly)


        # Intervals, markers and baseline extracted
        # -------------------------------------------------------
        noise_baseline,     noise_baseline_start,     noise_baseline_end,     noise_baseline_mean     = noise_intervals['noise_baseline']
        noise_anticipation, noise_anticipation_start, noise_anticipation_end, noise_anticipation_mean = noise_intervals['noise_anticipation']
        noise_task,         noise_task_start,         noise_task_end,         noise_task_mean         = noise_intervals['noise_task']
        noise_recovery,     noise_recovery_start,     noise_recovery_end,     noise_recovery_mean     = noise_intervals['noise_recovery']

        arith_baseline,     arith_baseline_start,     arith_baseline_end,     arith_baseline_mean     = arith_intervals['arith_baseline']
        arith_anticipation, arith_anticipation_start, arith_anticipation_end, arith_anticipation_mean = arith_intervals['arith_anticipation']
        arith_task,         arith_task_start,         arith_task_end,         arith_task_mean         = arith_intervals['arith_task']
        arith_recovery,     arith_recovery_start,     arith_recovery_end,     arith_recovery_mean     = arith_intervals['arith_recovery']


        ##############################################################################
        # Global metrics per participant
        ##############################################################################

        global_results = []

        # Check if arith data exists (adjust condition based on how missing data is represented)
        has_arith = arith_baseline_start is not None and not np.isnan(arith_baseline_start)

        # Define expected metrics (for NaN filling when data is missing)
        expected_metrics = [
            'mean_HR', 'mean_RRI', 'RMSSD', 'SDNN', 
            #'pNN50',
        ]

        # Build global_blocks based on data availability
        if has_arith:
            # Full data: noise + arith
            global_blocks = {
                "global": [
                    {"label": "global_all", "start": noise_baseline_start, "end": arith_recovery_end}
                ],
                "noise": [
                    {"label": "noise_total", "start": noise_baseline_start, "end": noise_recovery_end}
                ],
                "arith": [
                    {"label": "arith_total", "start": arith_baseline_start, "end": arith_recovery_end}
                ]
            }
        else:
            # Noise only: global = noise range, arith = NaN
            print("  ⚠ No arith data detected - using noise-only for global")
            global_blocks = {
                "global": [
                    {"label": "global_all", "start": noise_baseline_start, "end": noise_recovery_end}
                ],
                "noise": [
                    {"label": "noise_total", "start": noise_baseline_start, "end": noise_recovery_end}
                ],
                "arith": [
                    {"label": "arith_total", "start": None, "end": None}  # Will trigger NaN filling
                ]
            }

        # Loop through global blocks
        for block_name, segments in global_blocks.items():
            print(f"\nCalculating {block_name.upper()} metrics...")
            
            for segment in segments:
                label = segment["label"] # take out "task_" part in label
                start = segment["start"]
                end = segment["end"]
                
                # Check if segment has valid start/end
                if start is None or end is None:
                    # Fill with NaN values
                    print(f"  Processing {label}: No data - filling with NaN")
                    for metric_name in expected_metrics:
                        global_results.extend(
                            build_result_row(
                                participant_info,
                                task=block_name,
                                task_label=label,
                                rel_time=np.nan,
                                plot_time=np.nan,
                                metric_name=metric_name,
                                metric_value=np.nan,
                                baseline_mean=np.nan,
                                xdf_file=xdf_file,
                                #cont_time=np.nan
                            )
                        )
                    print(f"    ✓ {len(expected_metrics)} NaN metrics added")
                    continue  # Skip to next segment
                
                print(f"  Processing {label}: {start:.2f}s to {end:.2f}s")
                
                # Extract metrics for this segment
                segment_metrics = get_segment_metrics(
                    signal_clean,
                    start,
                    end,
                    fs_ppg,
                    filtering="physiological",
                    show=False
                )
                
                if segment_metrics:
                    for metric_name, metric_value in segment_metrics.items():
                        global_results.extend(
                            build_result_row(
                                participant_info,
                                task=block_name,
                                task_label=label,
                                rel_time=np.nan,
                                plot_time=np.nan,
                                metric_name=metric_name,
                                metric_value=metric_value,
                                baseline_mean=np.nan,
                                xdf_file=xdf_file,
                                #cont_time=np.nan
                            )
                        )
                    print(f"    ✓ {len(segment_metrics)} metrics extracted")
                else:
                    # Empty segment returned - fill with NaN
                    print(f"    ⚠ Empty segment - filling with NaN")
                    for metric_name in expected_metrics:
                        global_results.extend(
                            build_result_row(
                                participant_info,
                                task=block_name,
                                task_label=label,
                                rel_time=np.nan,
                                plot_time=np.nan,
                                metric_name=metric_name,
                                metric_value=np.nan,
                                baseline_mean=np.nan,
                                xdf_file=xdf_file,
                                #cont_time=np.nan
                            )
                        )
                    print(f"    ✓ {len(expected_metrics)} NaN metrics added")                


        #################################################
        # Compile results with continuous time interval
        #################################################
        results = []
        cont_time = 0

        # NOISE & ARITHMETIC BLOCK
        #------------------------------------------------------------    
        #dictionary containing the interval & corresponding start, end and rel_start
        blocks = {
            "noise": [
                {"label": "baseline", "start": noise_baseline_start, "end": noise_baseline_end, "rel_start": -270, "rel_end": -60},
                {"label": "anticipation", "start": noise_anticipation_start, "end": noise_anticipation_end, "rel_start": -30, "rel_end": -30},
                {"label": "task", "start": noise_task_start, "end": noise_task_end, "rel_start": 0, "rel_end": 270},
                {"label": "recovery", "start": noise_recovery_start, "end": noise_recovery_end, "rel_start": 300, "rel_end": 300}
            ],
            "arith": [
                {"label": "baseline", "start": arith_baseline_start, "end": arith_baseline_end, "rel_start": -270, "rel_end": -60},
                {"label": "anticipation", "start": arith_anticipation_start, "end": arith_anticipation_end, "rel_start": -30, "rel_end": -30},
                {"label": "task", "start": arith_task_start, "end": arith_task_end, "rel_start": 0, "rel_end": 270},
                {"label": "recovery", "start": arith_recovery_start, "end": arith_recovery_end, "rel_start": 300, "rel_end": 300}
            ]
        }

        for block_name, phases in blocks.items():
            baseline_metrics = None

            for phase in phases: 
                #define the variables based off blocks
                label = phase["label"]
                start = phase["start"]
                end = phase["end"]
                rel_start = phase["rel_start"]
                rel_end = phase["rel_end"]
                plot_start = rel_start + 15

                #handle baseline
                if "baseline" in label:
                    print(f"\nextracting metrics for {block_name} baseline: {start} to {end}")
                    baseline_metrics = get_segment_metrics(signal_clean, 
                                                           start, 
                                                           end, 
                                                           fs_ppg,
                                                           filtering='physiological', 
                                                           show=False
                                                           )
                    
                    for metric_name, metric_value in baseline_metrics.items():
                        #établir baseline selon metric
                        baseline_mean = baseline_metrics[metric_name]
                        #append results
                        results.extend(
                            build_result_row(
                                participant_info,
                                task=block_name,
                                task_label=label,
                                rel_time=rel_start,
                                plot_time = plot_start,
                                metric_name=metric_name, 
                                metric_value=metric_value,
                                baseline_mean=baseline_mean,
                                xdf_file=xdf_file,
                                #cont_time=cont_time
                            )
                        )
                    cont_time += 180
                    
                #handle anticipation/task/recovery
                else: 
                      for rel, seg in bin_segments(signal_clean, start, end, rel_start, rel_end):
                        seg_begin = seg["time_seconds"].values[0]
                        seg_end = seg["time_seconds"].values[-1]
                        plot = rel + 15
                        print(f"extracting metrics for {label} segment ({rel}): {seg_begin} to {seg_end}")

                        metrics = get_segment_metrics(
                            seg, 
                            seg_begin, 
                            seg_end,
                            fs_ppg,
                            filtering='physiological',
                            show=False
                        )

                        for metric_name, metric_value in metrics.items():
                            baseline_mean = baseline_metrics[metric_name]   
                            results.extend(
                                build_result_row(
                                    participant_info,
                                    task=block_name,
                                    task_label=label,
                                    rel_time=rel,
                                    plot_time = plot,
                                    metric_name=metric_name,
                                    metric_value=metric_value,
                                    baseline_mean=baseline_mean,
                                    xdf_file=xdf_file,
                                    #cont_time=cont_time
                                )
                            )
                        cont_time += 30

        print(f"Continuous HRV metrics calculated for {participant}")
        df = pd.DataFrame(results)
        df_global = pd.DataFrame(global_results)
        results.extend(global_results) 
        return results

    ########################################
    #7) error handling
    ########################################

    except Exception as e:
        # If ANY error occurs, return error information
        # Return a dict with status='FAILED' and the error message
        return [{
            "xdf_file": str(xdf_file.name),
            "status": "FAILED",
            "error": str(e),
            "task_moment": None,
            "time_interval_relative": None,
            "time_center_plot": None,
            "mean_hr": None,
            "diff_vs_baseline": None,
            "pct_change_vs_baseline": None,
            **extract_participant_info(xdf_file)
        }]



