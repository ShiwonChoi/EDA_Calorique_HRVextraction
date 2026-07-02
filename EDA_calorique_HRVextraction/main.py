import pandas as pd

from lib.config import get_participant_paths, OUTPUT_COLUMNS, DATA_DIR, output_file
from lib.CAL_process import full_process_single


# ============================================================================
# Batch processing
# ============================================================================

def batch_process_all(participants_dir=DATA_DIR):
    """
    Process all SC_* participant folders and return two consolidated
    DataFrames — one for temporal HRV (CAL_temp) and one for frequency
    HRV (CAL_freq) — covering the full cohort.

    Returns
    -------
    df_temp : pd.DataFrame  — all participants' temporal HRV rows.
    df_freq : pd.DataFrame  — all participants' frequency HRV rows.
    """
    participant_paths = get_participant_paths(root=participants_dir)

    print("=" * 60)
    print("BATCH PROCESSING STARTED")
    print(f"Participants found: {len(participant_paths)}")
    print("=" * 60)

    all_temp = []
    all_freq = []
    successful = 0
    failed     = 0

    processed = set()

    for i, p_path in enumerate(participant_paths, 1):

        if p_path.name in processed:
            print(f"[{i}/{len(participant_paths)}] SKIP: {p_path.name} (already processed)")
            continue

        print(f"\n[{i}/{len(participant_paths)}] Processing: {p_path.name} ...")

        try:
            _, df_temp, df_freq = full_process_single(p_path, 
                                                      use_physio=True, use_stat=False, 
                                                      show=False, 
                                                      bin=30)

            if df_temp.empty and df_freq.empty:
                print(f"  -> FAILED (empty output)")
                failed += 1
                continue

            all_temp.append(df_temp)
            all_freq.append(df_freq)
            processed.add(p_path.name)
            successful += 1
            print(f"  -> SUCCESS  ({len(df_temp)} temp rows, {len(df_freq)} freq rows)")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print("BATCH PROCESSING COMPLETED")
    print(f"Successful: {successful} | Failed: {failed}")
    print("=" * 60)

    df_temp_all = pd.concat(all_temp, ignore_index=True) if all_temp else pd.DataFrame(columns=OUTPUT_COLUMNS)
    df_freq_all = pd.concat(all_freq, ignore_index=True) if all_freq else pd.DataFrame(columns=OUTPUT_COLUMNS)

    return df_temp_all, df_freq_all


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":

    print(f"Data directory : {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found at {DATA_DIR}")

    df_temp, df_freq = batch_process_all()

    print("\nTemporal HRV results:")
    print(df_temp.head())

    print("\nFrequency HRV results:")
    print(df_freq.head())

    # Save to Results/ using output_file stem from config
    temp_out = output_file.with_name(output_file.stem + "_temp.csv")
    freq_out = output_file.with_name(output_file.stem + "_freq.csv")

    df_temp.to_csv(temp_out, index=False)
    df_freq.to_csv(freq_out, index=False)

    print(f"\n\nTemporal results saved to : {temp_out}")
    print(f"Frequency results saved to : {freq_out}")
