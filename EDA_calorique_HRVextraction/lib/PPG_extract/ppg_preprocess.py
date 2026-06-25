import json
import pandas as pd
from pathlib import Path
from lib.config import *


def match_events(participant_path):
    """
    Cross-check the events JSON and CSV files for every trial of a participant.

    For each matched pair (same P00#_Trial0#_condition_timestamp key) inside
    'Stress measures/', the function compares:

      1. First-sample anchor  — CSV comment "# First sample at: ..." vs
                                 JSON synchronization.first_sample_utc
      2. Event count          — number of rows in CSV vs entries in JSON events list
      3. Event-type sequence  — ordered list of event_type values
      4. Timestamps           — timestamp_utc per event (exact string comparison)

    Args:
        participant_path: Path to the SC_## participant folder.

    Returns:
        List of dicts, one per trial, with keys:
            participant      (str)
            trial            (str)
            condition        (str)
            status           ('OK' | 'MISMATCH' | 'MISSING_CSV' | 'MISSING_JSON')
            csv_count        (int | None)
            json_count       (int | None)
            mismatches       (list[str])  — empty when status == 'OK'
    """
    participant_path = Path(participant_path)
    stress_dir = participant_path / "Stress measures"

    if not stress_dir.exists():
        raise FileNotFoundError(
            f"No 'Stress measures' folder found in {participant_path}"
        )

    # Build lookup dicts: shared key → file path
    csv_lookup  = {
        f.stem[len("events_"):]: f
        for f in stress_dir.glob("events_*.csv")
    }
    json_lookup = {
        f.stem[len("events_"):]: f
        for f in stress_dir.glob("events_*.json")
    }

    all_keys = sorted(set(csv_lookup) | set(json_lookup))
    results  = []

    for key in all_keys:
        parts         = key.split("_")
        participant_id = parts[0]
        trial          = parts[1]
        condition      = "_".join(parts[2:-2])

        base = {"participant": participant_id, "trial": trial, "condition": condition}

        # ── Missing-pair guard ───────────────────────────────────────────────
        if key not in csv_lookup:
            results.append({**base, "status": "MISSING_CSV",
                            "csv_count": None, "json_count": None,
                            "mismatches": [f"No CSV found for key: {key}"]})
            continue

        if key not in json_lookup:
            results.append({**base, "status": "MISSING_JSON",
                            "csv_count": None, "json_count": None,
                            "mismatches": [f"No JSON found for key: {key}"]})
            continue

        mismatches = []

        # ── Load CSV ─────────────────────────────────────────────────────────
        df_csv = pd.read_csv(csv_lookup[key], comment="#")

        # Extract the first-sample timestamp from the comment line
        csv_first_sample = None
        with open(csv_lookup[key]) as fh:
            first_line = fh.readline().strip()
            if first_line.startswith("# First sample at:"):
                csv_first_sample = first_line.split("# First sample at:")[-1].strip()

        # ── Load JSON ────────────────────────────────────────────────────────
        with open(json_lookup[key]) as fh:
            jdata = json.load(fh)

        json_events      = jdata.get("events", [])
        json_first_sample = jdata.get("synchronization", {}).get("first_sample_utc")

        csv_count  = len(df_csv)
        json_count = len(json_events)

        # ── Check 1: first-sample anchor ─────────────────────────────────────
        if csv_first_sample != json_first_sample:
            mismatches.append(
                f"first_sample mismatch — CSV: '{csv_first_sample}' | "
                f"JSON: '{json_first_sample}'"
            )

        # ── Check 2: event count ─────────────────────────────────────────────
        if csv_count != json_count:
            mismatches.append(
                f"event count mismatch — CSV: {csv_count} | JSON: {json_count}"
            )

        # ── Check 3 & 4: event-type sequence and timestamps ──────────────────
        csv_types  = df_csv["event_type"].tolist()
        json_types = [ev["event_type"] for ev in json_events]

        if csv_types != json_types:
            # Report only the diverging positions to keep output compact
            diffs = [
                f"  row {i}: CSV='{c}' vs JSON='{j}'"
                for i, (c, j) in enumerate(
                    zip(csv_types, json_types), start=1
                )
                if c != j
            ]
            extras_csv  = csv_types[len(json_types):]
            extras_json = json_types[len(csv_types):]
            detail = "\n".join(diffs)
            if extras_csv:
                detail += f"\n  CSV has extra events: {extras_csv}"
            if extras_json:
                detail += f"\n  JSON has extra events: {extras_json}"
            mismatches.append(f"event_type sequence mismatch:\n{detail}")

        else:
            # Types match — check per-event timestamps
            ts_mismatches = []
            for i, (row, ev) in enumerate(
                zip(df_csv.itertuples(index=False), json_events), start=1
            ):
                csv_ts  = str(row.timestamp_utc).strip()
                json_ts = str(ev.get("timestamp_utc", "")).strip()
                if csv_ts != json_ts:
                    ts_mismatches.append(
                        f"  row {i} ({ev['event_type']}): "
                        f"CSV='{csv_ts}' | JSON='{json_ts}'"
                    )
            if ts_mismatches:
                mismatches.append(
                    "timestamp_utc mismatch:\n" + "\n".join(ts_mismatches)
                )

        status = "OK" if not mismatches else "MISMATCH"
        results.append({
            **base,
            "status":     status,
            "csv_count":  csv_count,
            "json_count": json_count,
            "mismatches": mismatches,
        })

    return results


def clean_events(df_events):
    """
    Retain only the event rows needed for analysis.

    Baseline trials  (condition == 'baseline') : keeps event_types listed in ALLBASECOND.
    Condition trials (RW, LW, RC, LC)          : keeps event_types listed in ALLSTIMCOND.
    """
    STIM_CONDITIONS = {'RW', 'LW', 'RC', 'LC'}
    baseline_types  = {t[0] for t in ALLBASECOND}
    stim_types      = {t[0] for t in ALLSTIMCOND}

    mask = (
        ((df_events['condition'] == 'baseline') & (df_events['event_type'].isin(baseline_types))) |
        (df_events['condition'].isin(STIM_CONDITIONS) & df_events['event_type'].isin(stim_types))
    )
    return df_events[mask].reset_index(drop=True)


def zero_reference(df_ppg, df_events, offset=0):
    df_ppg = df_ppg.copy()
    df_ppg["Time Stamp"] = (
        df_ppg.groupby("trial", sort=False)["Time Stamp"]
        .transform(lambda ts: ts - ts.iloc[0] + offset)
    )
    return df_ppg



def correct_time(df, df_events):

    if df.columns.shape[0] == 1:
        Time = (df.index[:] - df.index[0]) * 1000

        x = Time.values

        time_idx = pd.to_timedelta(Time, unit='ms')

        fs = 1 / (x[1] - x[0])
        df.index = time_idx

    else:
        Time = (df.iloc[:, 0] - df.iloc[0, 0]) * 1000

        x = Time.values
        time_idx = pd.to_timedelta(Time, unit='ms')

        fs = 1 / (x[1] - x[0]) # sampling rate (freq sample)
        df.index = time_idx

    Time_s = df.index.total_seconds()
    df.insert(1, 'time_seconds', Time_s)

    if df_events is not None:

        event_time = (df_events.iloc[:, 0] - df.iloc[0, 0]) * 1000

        x = event_time.values / 1000
        time_idx = pd.to_timedelta(event_time, unit='ms')

        df_events.index = time_idx

        Time_s = df_events.index.total_seconds()
        df_events.insert(1, 'time_seconds', Time_s)

    return df, df_events, fs


def crop_dataset(df_ppg, df_events):
    if df_events is not None and len(df_events) > 0:
        first_marker = df_events['time_since_connected_ms'].iloc[0]
        last_marker  = df_events['time_since_connected_ms'].iloc[-1]
        df_ppg = df_ppg.loc[
            (df_ppg['Time Stamp'] >= first_marker) &
            (df_ppg['Time Stamp'] <= last_marker)
        ]
    return df_ppg
