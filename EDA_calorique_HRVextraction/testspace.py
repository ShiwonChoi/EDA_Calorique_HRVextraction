from lib.config import PARTICIPANTS_DIR, PROCESSED_PPG_DIR, get_participant_paths
from lib.PPG_extract.load_and_clean_ppg import load_and_clean_ppg

# --- Single participant call ---
participant_path = PARTICIPANTS_DIR / "SC_01"
df_ppg, df_events, fs, badsegments = load_and_clean_ppg(participant_path, show=True)

print(df_ppg.head())
print(df_events.head())

# --- Loop over all discovered participants ---
# for participant_path in get_participant_paths():
#     df_ppg, df_events = load_ppg(participant_path, show=True)
#     out_path = PROCESSED_PPG_DIR / f"{participant_path.name}_ppg.csv"
#     df_ppg.to_csv(out_path, index=False)
#     print(f"Saved: {out_path}")
