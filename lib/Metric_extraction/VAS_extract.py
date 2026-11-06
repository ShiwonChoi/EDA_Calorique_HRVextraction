# VAS (subjective stress) extraction
#
# The touch_data_*.csv file is a continuous recording of a subjective-stress
# VAS rating (`position`, 0-1, scaled ×100 to a 0-100 score). 

import pandas as pd
from pathlib import Path


def load_touch_vas(participant_path):
    participant_path = Path(participant_path)
    touch_files = sorted(participant_path.glob("touch_data_*.csv"))
    if len(touch_files) != 1:
        print(f"  VAS: expected one touch_data_*.csv in {participant_path}, "
              f"found {len(touch_files)} — skipping VAS")
        return None

    df = pd.read_csv(touch_files[0])
    missing = {"monotonic_s", "position", "is_touching"} - set(df.columns)
    if missing:
        print(f"  VAS: {touch_files[0].name} missing {sorted(missing)} — skipping VAS")
        return None

    # monotonic_s is the clock shared with the event log; vas on a 0-100 scale
    out = pd.DataFrame({
        "monotonic_s": df["monotonic_s"].astype(float),
        "vas":         df["position"].astype(float) * 100.0,
        "is_touching": df["is_touching"],
    }).sort_values("monotonic_s").reset_index(drop=True)

    print(f"  VAS: {len(out)} samples loaded "
          f"[{out['monotonic_s'].min():.1f}s – {out['monotonic_s'].max():.1f}s]")
    return out
