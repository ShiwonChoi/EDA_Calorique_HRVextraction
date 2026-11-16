import pandas as pd

from lib.config import get_participant_paths, OUTPUT_COLUMNS, DATA_DIR, output_file
from lib.CAL_process import full_process_single


# ============================================================================
# Batch processing
# ============================================================================

def batch_process_all(participants_dir=DATA_DIR):
    participant_paths = get_participant_paths(root=participants_dir)

    print("=" * 60)
    print("BATCH PROCESSING STARTED")
    print(f"Participants found: {len(participant_paths)}")
    print("=" * 60)

    all_vas = []
    failed_participants = []

    for i, p_path in enumerate(participant_paths, 1):
        print(f"\n[{i}/{len(participant_paths)}] Processing: {p_path.name} ...")

        try:
            _, df_vas, df_sound = full_process_single(p_path, show=True)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed_participants.append(p_path.name)
            continue

        if df_sound.empty:
            print("  -> FAILED (no sounds matched to touch data)")
            failed_participants.append(p_path.name)
            continue

        all_vas.append(df_vas)
        n_missing = int(df_vas["Value"].isna().sum())
        print(f"  -> SUCCESS  ({len(df_sound)} sounds, {len(df_vas)} vas rows"
              + (f", {n_missing} missing values (see 'error')" if n_missing else "") + ")")

    print()
    print("=" * 60)
    print("BATCH PROCESSING COMPLETED")
    print(f"Successful: {len(all_vas)} | Failed: {len(failed_participants)}")
    if failed_participants:
        print(f"Failed participants: {', '.join(failed_participants)}")
    print("=" * 60)

    if not all_vas:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(all_vas, ignore_index=True)[OUTPUT_COLUMNS]


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
    df_vas.to_csv(output_file, index=False)

    print(f"\n\nVAS results saved to : {output_file}")
