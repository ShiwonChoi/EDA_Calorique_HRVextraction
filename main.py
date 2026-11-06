import pandas as pd

from lib.config import get_participant_paths, OUTPUT_COLUMNS, DATA_DIR, output_file
from lib.CAL_process import full_process_single


# ============================================================================
# Batch processing
# ============================================================================

def batch_process_all(participants_dir=DATA_DIR):
    """
    Process all judge-sound participant folders to return consolidated DataFrames : 
    subjective-stress VAS (CAL_vas)
    """
    participant_paths = get_participant_paths(root=participants_dir)

    print("=" * 60)
    print("BATCH PROCESSING STARTED")
    print(f"Participants found: {len(participant_paths)}")
    print("=" * 60)

    all_vas  = []
    successful = 0
    failed     = 0

    processed = set()

    for i, p_path in enumerate(participant_paths, 1):

        if p_path.name in processed:
            print(f"[{i}/{len(participant_paths)}] SKIP: {p_path.name} (already processed)")
            continue

        print(f"\n[{i}/{len(participant_paths)}] Processing: {p_path.name} ...")

        try:
            _, df_vas, df_sound = full_process_single(p_path, show=False)

            if df_sound.empty:
                print(f"  -> FAILED (no sounds matched)")
                failed += 1
                continue

            all_vas.append(df_vas)
            processed.add(p_path.name)
            successful += 1
            print(f"  -> SUCCESS  ({len(df_vas)} vas rows)")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print("BATCH PROCESSING COMPLETED")
    print(f"Successful: {successful} | Failed: {failed}")
    print("=" * 60)

    df_vas_all  = pd.concat(all_vas,  ignore_index=True) if all_vas  else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return df_vas_all


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":

    print(f"Data directory : {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found at {DATA_DIR}")

    df_vas = batch_process_all()

    print("\nVAS (subjective stress) results:")
    print(df_vas.head())

    # Save to Results/
    vas_out = output_file.with_name("processed_vas_results.csv")
    df_vas.to_csv(vas_out, index=False)

    print(f"\n\nVAS results saved to       : {vas_out}")
