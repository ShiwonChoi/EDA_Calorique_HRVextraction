"""
Quick test harness for lib.Plots.metric_boxplots.plot_total_metric_boxplot.

Points at one of the processed_ppg_results_*.csv files, picks a single metric
and value type, and draws the whole-task ('total') boxplots grouped by trial
and split by condition.
"""

from lib.config import output_dir
from lib.Plots_func.metric_boxplots import (
    plot_total_metric_boxplot,
    plot_total_metric_boxplot_by_trial,
)
from lib.Plots_func.interval_plots import (
    plot_intervals_by trial
)

# --- Pick which result file / metric / value type to plot ------------------
# Swap CSV_PATH between the temp / freq / gsr files and set METRIC to a metric
# that exists in that file:
#   temp : mean_HR, mean_RRI, RMSSD, SDNN
#   freq : VLF, LF, HF
#   gsr  : Tonic_SCL_mean, Tonic_SCL_slope, Phasic_AUC, Phasic_SCR_count, ...
CSV_PATH   = output_dir / "processed_ppg_results_temp.csv"
METRIC     = "mean_HR"
VALUE_TYPE = "raw"        # 'raw' | 'diff' | 'pct_change' | 'log_ratio'


if __name__ == "__main__":
    # One box per trial, all conditions (LC/LW/RC/RW/baseline) pooled together.
    plot_total_metric_boxplot_by_trial(
        csv_path=CSV_PATH,
        metric=METRIC,
        value_type=VALUE_TYPE,
        show=True,
    )

    # Condition-split version:
    # plot_total_metric_boxplot(
    #     csv_path=CSV_PATH, metric=METRIC, value_type=VALUE_TYPE, show=True,
    # )

# TODO plot interval measure
# TODO autocorrelation