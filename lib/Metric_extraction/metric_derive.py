import numpy as np

from lib.config import VAS_MODE_BIN_WIDTH


def _histogram_mode(v, bin_width=VAS_MODE_BIN_WIDTH):
    # center of the most populated bin (ties -> lowest bin)
    bins = np.floor(v / bin_width).astype(int)
    counts = np.bincount(bins - bins.min())
    return float((bins.min() + np.argmax(counts) + 0.5) * bin_width)


def get_vas_metrics(vas):
    """
    mean / mode / median / std of the VAS values (0-100) of one sound interval.

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
        "mode":   _histogram_mode(v),
        "median": float(np.median(v)),
        "std":    float(np.std(v, ddof=1)) if v.size >= 2 else float("nan"),
    }
    return metrics, int(v.size)
