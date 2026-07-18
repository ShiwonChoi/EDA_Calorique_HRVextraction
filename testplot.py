"""
Test harness for the interval time-series plots.

Edit the CONFIG block below to choose which results file, metric and value_type
to plot, then run from the project root:

    python testplot.py

One figure is produced per condition, overlaying the study groups (HC, T) with a
mean line and a shaded error band. Figures are written under Plots/<format>/ and
shown on screen.
"""

import pandas as pd

from lib.Plots_func.plotter import plot_all_task_all_group

# ============================== CONFIG ==============================
# Which results CSV to read.
#   Results/processed_ppg_results_temp.csv  -> RMSSD, SDNN, mean_HR, mean_RRI
#   Results/processed_ppg_results_freq.csv  -> LF, HF
#   Results/processed_gsr_results.csv       -> Tonic_SCL_*, Phasic_*
#   Results/processed_vas_results.csv       -> VAS_mean, VAS_median, VAS_std
CSV_FILE   = "Results/processed_ppg_results_freq.csv"

# Which metric (the `Metric` column) to plot.
METRIC     = "HF"

# Which value representation (the `Value_type` column):
#   'raw' | 'diff' | 'pct_change' | 'log_ratio'
VALUE_TYPE = "pct_change"

# Shaded band around the mean: 'sem' (±1 SEM) or 'ci' (t-distribution CI).
ERROR_BAND = "sem"
# ===================================================================


def main():
    df = pd.read_csv(CSV_FILE)
    figs = plot_all_task_all_group(
        df,
        metric=METRIC,
        value_type=VALUE_TYPE,
        error_band=ERROR_BAND,
        save=True,
        show=True,
    )
    print(f"\nProduced {len(figs)} figure(s) for "
          f"{METRIC} / {VALUE_TYPE} (band: {ERROR_BAND}).")


if __name__ == "__main__":
    main()
