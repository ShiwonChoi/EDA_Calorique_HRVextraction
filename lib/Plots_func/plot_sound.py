import re

import matplotlib.pyplot as plt
import numpy as np

from lib.Plots_func.plot_config import FIGURES_PATH
from lib.Plots_func.plot_utils import _save

SOUND_FIGURES_PATH = FIGURES_PATH / "sound_touch"
HIST_FIGURES_PATH = FIGURES_PATH / "sound_hist"


def _safe(name):
    """Make a string safe to use as a file/folder name."""
    return re.sub(r'[^\w\-.]+', '_', str(name)).strip('_')


SKIP_COLORS = {"sound_gap": "tab:gray", "task_pause": "tab:orange"}


def plot_sound_touch(touch, trial, category, sound_name, start_s, end_s,
                     skips=None, pad_s=3.0, segments=None, participant=None,
                     show=False, save=True):
    if not (show or save):
        return

    duration = end_s - start_s
    w0, w1 = -pad_s, duration + pad_s

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(0, duration, color="tab:blue", alpha=0.08, label="sound")
    if skips is not None and len(skips):
        labelled = set()
        for sk in skips.itertuples():
            a, b = max(sk.start - start_s, w0), min(sk.end - start_s, w1)
            if b <= a:
                continue
            ax.axvspan(a, b, color=SKIP_COLORS.get(sk.type, "tab:gray"), alpha=0.2,
                       label=None if sk.type in labelled else sk.type.replace("_", " "))
            labelled.add(sk.type)
    ax.plot(touch["t_rel_s"], touch["vas"], "-o", ms=2, lw=1, color="tab:blue")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(duration, color="r", ls="--", lw=1)
    if segments:
        _draw_segments(ax, segments)
    ax.set_xlim(w0, w1)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time from sound onset (s)")
    ax.set_ylabel("VAS position (0-100)")
    title = f"Trial {trial} - {category} / {sound_name}"
    if participant:
        title = f"{participant} - {title}"
    ax.set_title(title)
    fig.tight_layout()

    # Save BEFORE show: plt.show() can leave the figure empty once the window closes
    if save:
        parts = [participant, f"trial{int(trial):03d}", sound_name]
        filename = "_".join(_safe(p) for p in parts if p is not None)
        _save(fig, filename, SOUND_FIGURES_PATH / _safe(category))

    if show:
        plt.show()
    plt.close(fig)


def _draw_segments(ax, segments):
    """t_start / t_90 lines, plateaus (numbered) and the 90 % reach of each rise."""
    for key, color in (("t_start", "tab:green"), ("t_90", "tab:purple")):
        x = segments.get(key)
        if x is not None and np.isfinite(x):
            ax.axvline(x, color=color, ls=":", lw=1.5, label=f"{key} = {x:.2f}s")
    for k, p in enumerate(segments.get("plateaus", []), start=1):
        ax.hlines(p["level"], p["t0"], p["t1"], color="tab:red", lw=3, alpha=0.6,
                  label="plateau" if k == 1 else None)
        ax.annotate(str(k), ((p["t0"] + p["t1"]) / 2, p["level"]),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8, color="tab:red")
    for r in segments.get("rises", []):
        ax.plot(r["t_reach"], r["v_reach"], "v", color="tab:red", ms=6)


def _plot_histogram_mode(v, bin_width, mode, trial, category, sound_name,
                         participant=None, dt=None, show=True, save=True):
    if not (show or save):
        return

    v = np.asarray(v, dtype=float)
    lo = np.floor(v.min() / bin_width) * bin_width
    hi = (np.floor(v.max() / bin_width) + 1) * bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    if dt is not None and np.ndim(dt) == 1:      # per-sample durations -> dwell time
        heights, edges = np.histogram(v, bins=edges, weights=np.asarray(dt, float))
    else:
        counts, edges = np.histogram(v, bins=edges)
        heights = counts if dt is None else counts * dt
    mode_lo = np.floor(mode / bin_width) * bin_width
    colors = ["tab:red" if np.isclose(e, mode_lo) else "tab:blue" for e in edges[:-1]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(edges[:-1], heights, width=bin_width, align="edge", color=colors,
           edgecolor="white", lw=0.5)
    ax.axvline(mode, color="k", ls="--", lw=1, label=f"mode = {mode:.1f}")
    ax.set_xlim(min(lo, 0), max(hi, 100))
    ax.set_xlabel("VAS position (0-100)")
    ax.set_ylabel("Samples" if dt is None else "Dwell time (s)")
    title = f"Trial {trial} - {category} / {sound_name}"
    if participant:
        title = f"{participant} - {title}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save:
        parts = [participant, f"trial{int(trial):03d}", sound_name]
        filename = "_".join(_safe(p) for p in parts if p is not None)
        _save(fig, filename, HIST_FIGURES_PATH / _safe(category))

    if show:
        plt.show()
    plt.close(fig)
