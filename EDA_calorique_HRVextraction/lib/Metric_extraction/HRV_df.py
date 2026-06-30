import datetime as datetime
import math


def bin_segments(signal, start, end, rel_start, rel_end, interval=30):

    rows = []
    t = start
    rel = rel_start

    # Loop while rel < rel_end (rel_end is exclusive)
    while rel <= rel_end:
        next_t = min(t + interval, end)
        next_rel = rel + interval
        
        # Extract segment for this bin
        # Use < for upper bound to avoid overlap, except for the very last bin
        if next_rel < rel_end:
            # Not the last bin → half-open interval [t, next_t)
            seg = signal[
                (signal["time_seconds"] >= t) &
                (signal["time_seconds"] <  next_t)
            ].copy()
        else:
            # Last bin → inclusive on both ends [t, next_t]
            seg = signal[
                (signal["time_seconds"] >= t) &
                (signal["time_seconds"] <= next_t)
            ].copy()
        
        if len(seg) > 0:
            rows.append((rel, seg))
        
        t = next_t
        rel = next_rel

    return rows


def _derive_baseline_corrected(metric_value, baseline_mean, condition):
    """
    Compute diff / pct_change / log_ratio vs baseline.
    Baseline rows get 0.0 by convention (not NaN).
    """
    is_baseline = condition in ("baseline", "Trial00")

    if is_baseline:
        return {'diff': 0.0, 'pct_change': 0.0, 'log_ratio': 0.0}

    is_nan = math.isnan(metric_value) or math.isnan(baseline_mean)
    if is_nan:
        return {'diff': float('nan'), 'pct_change': float('nan'), 'log_ratio': float('nan')}

    diff = metric_value - baseline_mean
    pct_change = (diff / baseline_mean * 100) if baseline_mean != 0 else (0.0 if diff == 0 else float('nan'))
    if metric_value > 0 and baseline_mean > 0:
        log_ratio = math.log(metric_value / baseline_mean)
    else:
        log_ratio = float('nan')

    return {'diff': float(diff), 'pct_change': float(pct_change), 'log_ratio': float(log_ratio)}


def build_result_row(
        participant_id, trial, condition, rel_time, abs_time, metric_name, metric_value,
        baseline_mean,
        status="SUCCESS", error=None):
    
    derived = _derive_baseline_corrected(metric_value, baseline_mean, condition)

    base = {
        "participant":            participant_id,
        "trial":                  trial,
        "condition":              condition,
        "rel_time (ms)":          rel_time, 
        "abs_time (ms)":          abs_time, 
        "Metric":                 metric_name,
        "status":                 status,
        "error":                  error,
    }

    value_map = {
        "raw":        float(metric_value),
        "diff":       derived['diff'],
        "pct_change": derived['pct_change'],
        "log_ratio":  derived['log_ratio'],
    }

    return [
        {**base, "Value_type": vt, "Value": val}
        for vt, val in value_map.items()
    ]

