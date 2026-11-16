import re

import numpy as np

from lib.config import (
    VAS_MODE_BIN_WIDTH, VAS_IMAX_EXCLUDE_S, VAS_ONSET_FRAC, VAS_RISE_FRAC,
    VAS_PLATEAU_MIN_S, VAS_NO_RESPONSE, VAS_PEAK_MARGIN, VAS_SPIKE_MAX_S,
    VAS_JUMP_MAX_S, VAS_MIN_TOUCH_FRAC, VAS_METRIC_UNITS,
    VAS_REST_MAX, VAS_REST_CAP_FRAC,
)
from lib.Plots_func.plot_sound import _plot_histogram_mode


def _histogram_mode(v, bin_width=VAS_MODE_BIN_WIDTH, plot=True, **plot_kwargs):
    bins = np.floor(v / bin_width).astype(int)
    counts = np.bincount(bins - bins.min())
    mode_bin = bins.min() + np.argmax(counts)
    mode = float(np.median(v[bins == mode_bin]))
    if plot and plot_kwargs:
        _plot_histogram_mode(v, bin_width, mode, **plot_kwargs)
    return mode


def metric_unit(name):
    """Unit of a metric; per-plateau metrics (rise_dur_2, ...) use their base name."""
    base = re.sub(r"_\d+$", "", str(name))
    return VAS_METRIC_UNITS.get(name, VAS_METRIC_UNITS.get(base, ""))


def get_vas_metrics(vas):
    """
    mean / mode / median / std of the VAS values (0-100) of one sound interval.
    (Superseded by get_vas_phase_metrics; kept for reference.)

    Returns
    -------
    metrics : dict — mean, mode, median, std (std is NaN with <2 samples).
              Empty dict if there are no samples.
    n       : int  — number of VAS samples.
    """
    v = np.asarray(vas, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 1:
        return {}, 0

    metrics = {
        "mean":   float(np.mean(v)),
        "mode":   _histogram_mode(v, plot=False),
        "median": float(np.median(v)),
        "std":    float(np.std(v, ddof=1)) if v.size >= 2 else float("nan"),
    }
    return metrics, int(v.size)


# ============================================================================
# Three-phase VAS metrics: latency -> adjustment -> judgement
# ============================================================================

# fixed order of the status flags in the 'status' column
FLAG_ORDER = ["NO_T90", "NO_PLATEAU", "MULTI_PLATEAU", "SPIKE", "OVERSHOOT", "JUMP"]


def _peak_excursion_dur(t, v, i_peak, thr):
    """Duration (s) of the contiguous run around i_peak where v > thr."""
    above = v > thr
    a = b = i_peak
    while a > 0 and above[a - 1]:
        a -= 1
    while b < len(v) - 1 and above[b + 1]:
        b += 1
    # run lasts until the first sample back below thr (or the last sample)
    return float(t[min(b + 1, len(t) - 1)] - t[a])


def _upward_crossings(v, thr):
    """Indices i where v[i] > thr and v[i-1] <= thr."""
    return np.flatnonzero((v[1:] > thr) & (v[:-1] <= thr)) + 1


def _first(mask, i_from=0):
    """First index >= i_from where mask is True, or None."""
    idx = np.flatnonzero(mask[i_from:])
    return int(i_from + idx[0]) if idx.size else None


def _plateaus(t, v, i_from, min_level, bin_width=VAS_MODE_BIN_WIDTH,
              min_s=VAS_PLATEAU_MIN_S):
    """
    Plateaus = runs of consecutive samples in the same histogram bin lasting
    >= min_s, starting at or after index i_from, with a level above min_level.
    A run lasts from its first sample to the first sample of the next run.
    Consecutive plateaus in the same bin are merged.

    Returns a list of dicts: bin, level (median VAS), i0, i1 (exclusive),
    t0, t1, dur (summed duration of the runs, without excursions in between).
    """
    b = np.floor(v / bin_width).astype(int)
    edges = np.r_[0, np.flatnonzero(np.diff(b)) + 1, len(b)]
    out = []
    for a, e in zip(edges[:-1], edges[1:]):
        if a < i_from:
            continue
        t_end = t[e] if e < len(t) else t[-1]
        run_dur = t_end - t[a]
        if run_dur < min_s or np.median(v[a:e]) <= min_level:
            continue
        if out and out[-1]["bin"] == b[a]:
            out[-1].update(i1=e, t1=t_end)
            out[-1]["dur"] += run_dur
        else:
            out.append(dict(bin=int(b[a]), i0=int(a), i1=int(e),
                            t0=float(t[a]), t1=float(t_end), dur=float(run_dur)))
    for p in out:
        p["level"] = float(np.median(v[p["i0"]:p["i1"]]))
    return out


def _step_reach(v, i_from, level_from, level_to, frac=VAS_RISE_FRAC):
    """First index >= i_from where v has covered `frac` of the step (up or down)."""
    target = level_from + frac * (level_to - level_from)
    mask = v >= target if level_to >= level_from else v <= target
    return _first(mask, i_from)


def _rate(t, v, i0, i1):
    """
    (duration, rate, is_jump). A rise faster than VAS_JUMP_MAX_S is a jump
    (the participant tapped a new position): its rate is NaN.
    """
    dur = float(t[i1] - t[i0])
    if dur < VAS_JUMP_MAX_S:
        return dur, np.nan, True
    return dur, float((v[i1] - v[i0]) / dur), False


def _zero_judgement(t, v, late, metrics, segments, hist_kwargs):
    """
    Near-0 sound with the finger on the slider: a genuine judgement of 0
    (valid measure, e.g. relaxing sounds - not an error).
    Plateau metrics are filled from the samples after VAS_IMAX_EXCLUDE_S;
    every other metric (timing / rise) is set to 0.
    """
    tl, vl = t[late], v[late]
    level = float(np.median(vl))
    # time each sample lasts (until the next sample); robust to irregular sampling
    sample_dur = np.diff(tl, append=tl[-1])
    metrics.update(
        plateau_level=level,
        plateau_dur=float(sample_dur[vl < VAS_NO_RESPONSE].sum()),
        plateau_std=float(np.std(vl, ddof=1)) if vl.size >= 2 else np.nan,
        n_plateaus=1, n_changes=0, n_up=0, n_down=0,
        plateau_level_1=level,
    )
    for name, value in metrics.items():  # t_start, t_90, rise_* -> 0
        if not np.isfinite(value):
            metrics[name] = 0.0
    segments["plateaus"] = [dict(level=level, t0=float(tl[0]), t1=float(tl[-1]))]
    if hist_kwargs:  # histogram plot only; the level is the median, not the mode
        _histogram_mode(vl, plot=True, **{**hist_kwargs, "dt": sample_dur})


def _finish(metrics, errors, default, status, n, segments):
    """Give every NaN metric an error message (specific one if set, else default)."""
    for name, value in metrics.items():
        if not np.isfinite(value) and not errors.get(name):
            errors[name] = default
    return metrics, status, n, segments, errors


def get_vas_phase_metrics(t, vas, touching=None, duration=None, plot_kwargs=None):
    t = np.asarray(t, dtype=float)
    v = np.asarray(vas, dtype=float)
    n_raw = v.size
    keep = ~np.isnan(v)
    touched = keep if touching is None else (np.asarray(touching) == 1)
    touch_frac = float(touched.mean()) if n_raw else 0.0
    dur = float(duration) if duration is not None else (float(t[-1]) if n_raw else np.nan)
    t, v = t[keep], v[keep]
    n = int(v.size)
    nan = np.nan
    segments = {"t_start": nan, "t_90": nan, "plateaus": [], "rises": []}
    errors = {}

    # plateau 1 metrics are always reported so every sound has the same core rows
    metrics = dict(i_max=nan, t_start=nan, t_90=nan, rise_dur=nan, rise_rate=nan,
                   n_plateaus=nan, n_changes=nan, n_up=nan, n_down=nan,
                   plateau_level=nan, plateau_dur=nan, plateau_std=nan,
                   plateau_level_1=nan, rise_dur_1=nan, rise_rate_1=nan)

    if n_raw < 2:
        return _finish(metrics, errors, "no VAS samples recorded during the sound",
                       "NO_DATA", n, segments)
    if n < 2:
        return _finish(metrics, errors, "VAS empty during the sound (slider never touched)",
                       "NO_RESPONSE", n, segments)

    hist_kwargs = dict(plot_kwargs or {})

    # ---- i_max and thresholds (first VAS_IMAX_EXCLUDE_S s excluded) ----------
    late = t >= VAS_IMAX_EXCLUDE_S
    if not late.any():
        return _finish(metrics, errors,
                       f"sound too short ({dur:.1f} s): no samples after the "
                       f"{VAS_IMAX_EXCLUDE_S:g} s excluded from i_max",
                       "TOO_SHORT", n, segments)
    i_max = float(v[late].max())
    metrics["i_max"] = i_max
    if i_max < VAS_NO_RESPONSE:
        # near 0 throughout: genuine 0 judgement if the finger was on the slider
        if touch_frac < VAS_MIN_TOUCH_FRAC:
            return _finish(metrics, errors,
                           f"no response: finger on slider only {touch_frac:.0%} of the "
                           f"sound (< {VAS_MIN_TOUCH_FRAC:.0%})",
                           "NO_RESPONSE", n, segments)
        _zero_judgement(t, v, late, metrics, segments, hist_kwargs)
        return metrics, "ZERO_JUDGEMENT", n, segments, errors
    # onset threshold: 5 % of i_max, but at least the resting tolerance (participants
    # often rest the slider at ~4-6, not 0), capped at half of i_max for low ratings
    thr_lo = max(VAS_ONSET_FRAC * i_max, min(VAS_REST_MAX, VAS_REST_CAP_FRAC * i_max))
    thr_hi = VAS_RISE_FRAC * i_max
    flags = []

    # ---- phase 1: t_start = last upward crossing of thr_lo before t_90 -------
    ups = _upward_crossings(v, thr_lo)
    if ups.size == 0:
        return _finish(metrics, errors,
                       f"carry-over: VAS never dropped to the onset threshold ({thr_lo:.1f}), "
                       f"so no rise onset (t_start) could be found",
                       "CARRY_OVER", n, segments)
    i90 = _first(v >= thr_hi, ups[0])
    if i90 is not None:
        i_start = int(ups[ups <= i90][-1])
    else:
        i_start = int(ups[0])
        flags.append("NO_T90")
    t_start = float(t[i_start])
    metrics["t_start"] = segments["t_start"] = t_start
    left = dur - t_start  # time available after the rise onset

    # ---- phase 2: overall rise (i_max based) --------------------------------
    if i90 is not None:
        metrics["t_90"] = segments["t_90"] = float(t[i90])
        metrics["rise_dur"], metrics["rise_rate"], jump = _rate(t, v, i_start, i90)
        if jump:
            flags.append("JUMP")
            errors["rise_rate"] = (f"jump: rise took {metrics['rise_dur'] * 1000:.0f} ms "
                                   f"(< {VAS_JUMP_MAX_S * 1000:.0f} ms, slider tapped), rate undefined")
    else:
        msg = (f"VAS never reached 90 % of i_max ({thr_hi:.1f}) between t_start "
               f"({t_start:.1f} s) and sound end ({dur:.1f} s)")
        for name in ("t_90", "rise_dur", "rise_rate", "plateau_std"):
            errors[name] = msg

    # ---- phase 2: plateaus and per-plateau rises ----------------------------
    pl = _plateaus(t, v, i_start, thr_lo)
    segments["plateaus"] = pl
    levels = [p["level"] for p in pl]
    steps = np.diff(levels)
    metrics.update(n_plateaus=len(pl), n_changes=max(len(pl) - 1, 0),
                   n_up=int((steps > 0).sum()), n_down=int((steps < 0).sum()))
    if not pl:
        flags.append("NO_PLATEAU")
        if left < VAS_PLATEAU_MIN_S:
            msg = (f"no plateau possible: only {left:.1f} s between t_start ({t_start:.1f} s) "
                   f"and sound end ({dur:.1f} s), < {VAS_PLATEAU_MIN_S:g} s minimum")
        else:
            msg = (f"no plateau: no value held >= {VAS_PLATEAU_MIN_S:g} s between t_start "
                   f"({t_start:.1f} s) and sound end ({dur:.1f} s)")
        for name in ("n_plateaus", "plateau_dur", "plateau_level_1", "rise_dur_1", "rise_rate_1"):
            errors[name] = msg
    elif len(pl) >= 2:
        flags.append("MULTI_PLATEAU")
    # i_max clearly above every held level: brief SPIKE vs sustained OVERSHOOT
    peak_thr = max(levels) + VAS_PEAK_MARGIN if pl else np.inf
    if i_max > peak_thr:
        i_late = np.flatnonzero(late)
        i_peak = int(i_late[np.argmax(v[late])])
        peak_dur = _peak_excursion_dur(t, v, i_peak, peak_thr)
        flags.append("SPIKE" if peak_dur < VAS_SPIKE_MAX_S else "OVERSHOOT")

    for k, p in enumerate(pl, start=1):
        if k == 1:
            i_from, lv_from = i_start, 0.0
        else:
            i_from, lv_from = pl[k - 2]["i1"] - 1, pl[k - 2]["level"]
        i_reach = _step_reach(v, i_from, lv_from, p["level"])
        metrics[f"plateau_level_{k}"] = p["level"]
        if i_reach is None:
            metrics[f"rise_dur_{k}"] = metrics[f"rise_rate_{k}"] = nan
            errors[f"rise_dur_{k}"] = errors[f"rise_rate_{k}"] = (
                f"VAS never covered 90 % of the step to plateau {k} ({p['level']:.1f})")
            continue
        metrics[f"rise_dur_{k}"], metrics[f"rise_rate_{k}"], jump = _rate(t, v, i_from, i_reach)
        if jump:
            if "JUMP" not in flags:
                flags.append("JUMP")
            errors[f"rise_rate_{k}"] = (
                f"jump: rise to plateau {k} took {metrics[f'rise_dur_{k}'] * 1000:.0f} ms "
                f"(< {VAS_JUMP_MAX_S * 1000:.0f} ms, slider tapped), rate undefined")
        segments["rises"].append(dict(k=k, t_from=float(t[i_from]), t_reach=float(t[i_reach]),
                                      v_reach=float(v[i_reach])))

    # ---- phase 3: plateau level (mode from t_start), duration, std ----------
    if hist_kwargs:  # per-sample durations (timestamps can repeat) -> dwell time
        hist_kwargs.setdefault("dt", np.diff(t[i_start:], append=t[-1]))
    mode = _histogram_mode(v[i_start:], plot=bool(hist_kwargs), **hist_kwargs)
    mode_bin = int(np.floor(mode / VAS_MODE_BIN_WIDTH))
    metrics["plateau_level"] = mode
    metrics["plateau_dur"] = float(sum(p["dur"] for p in pl if p["bin"] == mode_bin))
    if i90 is not None:
        if n - i90 >= 2:
            metrics["plateau_std"] = float(np.std(v[i90:], ddof=1))
        else:
            errors["plateau_std"] = "fewer than 2 samples between t_90 and sound end"

    flags = [f for f in FLAG_ORDER if f in flags]
    return _finish(metrics, errors, "not computable", ";".join(flags) or "OK", n, segments)

