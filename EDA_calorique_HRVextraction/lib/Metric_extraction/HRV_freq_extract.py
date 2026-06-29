def get_segment_metrics(signal, start_time, end_time, sampling_rate, filtering='both', show=False):

    # Create mask for time range
    start_mask = start_time
    end_mask = end_time
    mask = (signal["time_seconds"] >= start_mask) & (signal["time_seconds"] <= end_mask)

    # Extract PPG data for this segment
    signal_extract = signal.loc[mask].copy()
    if len(signal_extract) == 0: 
        return {}
    
    #RRI/NN intervals (RR interval in ms = 60 000/Heart Rate or PPG_Rate_Corr -> HR = beats per minute so 60 000 ms = 1 min)
    rri_from_pulse_rate = 60000 / signal_extract['PPG_Rate_Corr']
    signal_extract.loc[:,'RRI_Intervals'] = rri_from_pulse_rate

    rri = signal_extract["RRI_Intervals"]
    valid_mask = pd.Series(True, index=signal_extract.index)
    threshold = None

    # RRI per segment
    # -------------------------------------------------------
    median = rri.median() # find median rr interval
    rris_per_sec = 1000 / median 
    expected_rris_30s = rris_per_sec * 30 # calculate how many of those intervals would be in a 30s period
    calculated_min_intervals = 0.5 * expected_rris_30s # the minimum intervals should be a percentage of that (start with 50%)
    min_intervals = max(calculated_min_intervals, 10) # what this means: never go lower than 10

    # PPG peaks per segment
    # ----------------------------------------------------------
    # What is the expected peaks per 30s segment
    peaks = signal_extract["PPG_Peaks_Corr"]
    peak_times = signal_extract.loc[peaks == 1, "time_seconds"]

    # inter-peak interval
    ipi = peak_times.diff().dropna()
    median_ipi = ipi.median()

    # Median peaks per second
    peaks_per_sec = 1 / median_ipi
    expected_peaks_30s = peaks_per_sec * 30
    calculated_min_peaks = 0.6 * expected_peaks_30s
    min_peaks = max(calculated_min_peaks, 15)

    # Number of peaks in the actual segment
    peak_count = (peaks == 1).sum()

    ##################################################
    # Filtering
    ##################################################
    
    #failsafe if not enough values within the interval in the first place
    if peak_count < min_peaks or np.isnan(min_peaks):
        print(f"    Segment [{start_time} - {end_time}] has too few PPG_Peaks ({peak_count} vs {min_peaks}) before filtering")
        return {'mean_HR': np.nan, 
                'std_HR': np.nan, 
                'mean_RRI': np.nan,
                'mean_NN': np.nan, 
                'RMSSD': np.nan, 
                'SDNN': np.nan, 
                #'pNN50': np.nan, 
            }

    # Physiological filtering (300–2000 ms)
    if filtering in "physiological":
        physio_mask = (rri >= 300) & (rri <= 2000)
        valid_mask = physio_mask

    # Statistical filtering (SD-based)
    elif filtering in "statistical":
        mean = rri.mean()
        sd = rri.std()
        #SD filtering via std dev
        sd_thresholds = [mean-3*sd,mean+3*sd]

        #MAD filtering via difference from median
        mad = np.median(np.abs(rri - median))
        mad_thresholds = [median-3*mad, median+3*mad]
     
        thresholds = mad_thresholds
        stat_mask = (rri >= thresholds[0]) & (rri <= thresholds[1])
        valid_mask = stat_mask
    
    elif filtering in "both":
        # Step 1: Physiological filter
        physio_mask = (rri >= 300) & (rri <= 2000)
        
        # Step 2: Statistical filter (calculated on physiologically-filtered data)
        rri_physio_filtered = rri[physio_mask]
        median = rri_physio_filtered.median()
        mad = np.median(np.abs(rri_physio_filtered - median))
        thresholds = [median - 3*mad, median + 3*mad]
        
        stat_mask = (rri >= thresholds[0]) & (rri <= thresholds[1])
        
        # Combine both masks
        valid_mask = physio_mask & stat_mask

    rri_physio_filtered = rri[valid_mask]

    # Count of rejected peaks and those that are kept
    nb_filtered_peaks  = len(rri_physio_filtered)
    nb_total_peaks     = len(rri)
    nb_rejected_peaks  = nb_total_peaks - nb_filtered_peaks
    print(f"    Segment [{start_time} - {end_time}] has {nb_rejected_peaks} rejected peaks out of {nb_total_peaks} total peaks")

    if len(rri_physio_filtered) < min_intervals:
        print(f"    Segment [{start_time} - {end_time}] has insufficient physiologically & statistically valid RRIs")
        return {'mean_HR': np.nan, 
                'std_HR': np.nan, 
                'mean_RRI': np.nan, 
                'mean_NN': np.nan,
                'RMSSD': np.nan, 
                'SDNN': np.nan, 
                #'pNN50': np.nan, 
            }


    ##################################################
    # Visualization (before applying filter)
    ##################################################
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left plot: Original RRI distribution
        axes[0].hist(rri, 50, facecolor='green', alpha=0.7, edgecolor='black')
        axes[0].set_title(f'Original RRI (n={len(rri)})')
        axes[0].set_xlabel('RRI (ms)')
        axes[0].set_ylabel('Count')
        
        # Add threshold lines if statistical filtering
        if filtering in ["statistical", "both"]:
            axes[0].axvline(thresholds[0], color='red', linestyle='--', lw=2, label=f'Lower: {thresholds[0]:.1f}')
            axes[0].axvline(thresholds[1], color='red', linestyle='--', lw=2, label=f'Upper: {thresholds[1]:.1f}')
        
        # Add physiological bounds if physiological filtering
        if filtering in ["physiological", "both"]:
            axes[0].axvline(300, color='blue', linestyle=':', lw=2, label='Physio: 300')
            axes[0].axvline(2000, color='blue', linestyle=':', lw=2, label='Physio: 2000')
        
        axes[0].legend()
        
        # Right plot: Filtered RRI distribution
        rri_filtered = rri[valid_mask]
        axes[1].hist(rri_filtered, 50, facecolor='red', alpha=0.7, edgecolor='black')
        axes[1].set_title(f'Filtered RRI (n={len(rri_filtered)}, removed={len(rri)-len(rri_filtered)})')
        axes[1].set_xlabel('RRI (ms)')
        axes[1].set_ylabel('Count')
        
        plt.suptitle(f'RRI Filtering: {filtering}')
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        plt.close()

    ##################################################
    # Apply filter to signal_extract
    ##################################################
    signal_filtered = signal_extract[valid_mask].copy()
    peaks_filtered = signal_filtered["PPG_Peaks_Corr"].values
    peak_count_filtered = (peaks_filtered == 1).sum()
    peak_times = signal_filtered.loc[signal_extract["PPG_Peaks_Corr"] == 1, "time_seconds"]

    # If too few intervals remain, return empty
    if peak_count_filtered >= min_peaks:

        #############################################################
        # General metrics
        #############################################################
        mean_hr = signal_filtered['PPG_Rate_Corr'].mean()
        mean_rri = signal_filtered['RRI_Intervals'].mean()
        nni = peak_times.diff().dropna() * 1000
        mean_nni = nni.mean()
        std_hr = signal_filtered['PPG_Rate_Corr'].std()
        duration = signal_filtered["time_seconds"].iloc[-1] - signal_filtered["time_seconds"].iloc[0]

        #############################################################
        # Time-Domain metrics
        #############################################################
        NNdiff = np.diff(signal_filtered['RRI_Intervals'])
        N = len(NNdiff)

        RMSSD = np.sqrt(np.mean(NNdiff ** 2))
        SDNN = np.std(signal_filtered['RRI_Intervals'], ddof=1)
        NN50 = np.sum(np.abs(NNdiff) > 50)
        pNN50 = NN50 / N * 100


        metrics = {
            'mean_HR': mean_hr,
            'std_HR': std_hr,
            'mean_RRI': mean_rri,
            'mean_NN': mean_nni,
            'RMSSD': RMSSD,
            'SDNN': SDNN,
            #'pNN50': pNN50,
        }

    else:
        print(f"    Segment [{start_time} - {end_time}] has insufficient RRIs : number of filtered peaks ({peak_count_filtered} vs {min_peaks}) was too low")

        metrics = {
            'mean_HR': np.nan,
            'std_HR': np.nan,
            'mean_RRI': np.nan,
            'mean_NN': np.nan,
            'RMSSD': np.nan,
            'SDNN': np.nan,
            #'pNN50': np.nan,
        }

    return metrics