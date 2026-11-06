import numpy as np
import pandas as pd

from lib.config import MAX_TASK_TRIALS


def match_sound_to_touch(df_events, df_touch):
    # task sounds only (training sounds are block == 'training')
    plays = df_events[
        (df_events["block"] == "task")
        & df_events["event_label"].isin(["sound_play_start", "sound_play_end"])
    ]

    starts = plays[plays["event_label"] == "sound_play_start"].set_index("trial")
    ends = plays[plays["event_label"] == "sound_play_end"].set_index("trial")
    if starts.index.has_duplicates or ends.index.has_duplicates:
        raise ValueError("Duplicate sound_play_start / sound_play_end for a trial")
    if not starts.index.equals(ends.index) and set(starts.index) != set(ends.index):
        raise ValueError("sound_play_start and sound_play_end trials do not match")

    out = pd.DataFrame({
        "trial":    starts.index.astype(int),
        "detail":   starts["detail"].to_numpy(),
        "start_s":  starts["monotonic_s"].to_numpy(float),
        "end_s":    ends.loc[starts.index, "monotonic_s"].to_numpy(float),
    }).sort_values("trial").reset_index(drop=True)

    if len(out) != MAX_TASK_TRIALS:
        print(f"  WARNING: {len(out)} task sounds found, expected {MAX_TASK_TRIALS}")
    if (out["end_s"] <= out["start_s"]).any():
        raise ValueError("A sound ends before it starts")

    # "Category/Sound.wav" -> category, sound_name
    parts = out.pop("detail").str.split("/", n=1, expand=True)
    out.insert(1, "category", parts[0])
    out.insert(2, "sound_name", parts[1])
    out["duration_s"] = out["end_s"] - out["start_s"]

    # touch samples per sound: [start, end), last sound closed at its end
    df_touch = df_touch.sort_values("monotonic_s").reset_index(drop=True)
    t = df_touch["monotonic_s"].to_numpy(float)
    lo = np.searchsorted(t, out["start_s"].to_numpy(), side="left")
    hi = np.searchsorted(t, out["end_s"].to_numpy(), side="left")
    hi[-1] = np.searchsorted(t, out["end_s"].iloc[-1], side="right")

    touch_slices = []
    for start_s, a, b in zip(out["start_s"], lo, hi):
        sl = df_touch.iloc[a:b].copy()
        sl["t_rel_s"] = sl["monotonic_s"] - start_s
        touch_slices.append(sl.reset_index(drop=True))
    out["touch"] = touch_slices
    out["n_samples"] = hi - lo

    return out[["trial", "category", "sound_name", "start_s", "end_s",
                "duration_s", "n_samples", "touch"]]
