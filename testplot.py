"""
Test harness for the interval time-series plots.

Edit the CONFIG block below to choose which results file, metric and value_type
to plot, then run from the project root:

    python testplot.py

Two plots are available (see `main`):
  * `baseline_comparison`      -- baseline reference vs recovery, HC vs T, in one
                                  figure (currently active).
  * `plot_all_task_all_group`  -- one figure per condition, groups overlaid
                                  across the full task+recovery axis (commented
                                  out below).
Figures are written under Plots/<format>/ and shown on screen.
"""

import pandas as pd

from lib.Plots_func.plotter import (
    baseline_comparison,
    baseline_comparison_by_condition,
    plot_all_task_all_group,
)

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
VALUE_TYPE = "raw"

# Shaded band around the mean: 'sem' (±1 SEM) or 'ci' (t-distribution CI).
ERROR_BAND = "sem"
# ===================================================================


def main():
    df = pd.read_csv(CSV_FILE)

    fig = baseline_comparison(
        df,
        metric=METRIC,
        value_type=VALUE_TYPE,
        error_band=ERROR_BAND,
        save=True,
        show=True,
    )
    status = "1 figure" if fig is not None else "no figure (no data)"
    print(f"\nProduced {status} for "
          f"{METRIC} / {VALUE_TYPE} (band: {ERROR_BAND}).")

    # Baseline vs a separate recovery curve per condition, each drawn in a
    # lightness variant of its group colour. Uncomment to run.
    baseline_comparison_by_condition(
        df,
        metric=METRIC,
        value_type=VALUE_TYPE,
        error_band=ERROR_BAND,
        save=True,
        show=True,
    )

    # One figure per condition, groups overlaid across the full task+recovery
    # axis. Uncomment to run instead of / alongside the baseline comparison.
    # figs = plot_all_task_all_group(
    #     df,
    #     metric=METRIC,
    #     value_type=VALUE_TYPE,
    #     error_band=ERROR_BAND,
    #     save=True,
    #     show=True,
    # )
    # print(f"Produced {len(figs)} figure(s) for "
    #       f"{METRIC} / {VALUE_TYPE} (band: {ERROR_BAND}).")


if __name__ == "__main__":
    main()
