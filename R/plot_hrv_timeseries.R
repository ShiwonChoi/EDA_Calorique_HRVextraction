# =============================================================================
#  plot_hrv_timeseries.R
#
#  A from-scratch R re-implementation of the Python function
#  `plot_all_task_all_group` (lib/Plots_func/plotter.py).
#
#  GOAL OF THE PLOT
#  ----------------
#  For one metric (e.g. RMSSD) we want a line chart where:
#     * the x-axis is *within-trial time* (seconds since the trial started),
#     * the y-axis is the mean of that metric across all participants,
#     * one coloured line per study group (HC = healthy controls, T = tinnitus),
#     * a shaded band around each line showing the uncertainty of the mean,
#     * one figure per experimental condition.
#
#  This file is written as a TEACHING SCRIPT for someone new to R. Every block
#  is explained: what it does, why it exists, and how it moves us one step
#  closer to the final plot. Read it top to bottom like a tutorial. The very
#  bottom (see "RUN IT" section) is where you actually call everything.
#
#  The pipeline has four stages, mirroring the Python code:
#     1. IMPORT   -> read the results CSV into a data frame.
#     2. FILTER   -> keep only the rows we want (one metric, interval rows).
#     3. AGGREGATE-> average across participants and build an error band.
#     4. PLOT     -> draw lines + bands + event markers, one figure per
#                    condition, and save to disk.
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 0 — LOAD THE TOOLBOXES ("packages")
# -----------------------------------------------------------------------------
# R ships with basic functions, but for data wrangling and plotting we lean on
# two famous add-on packages:
#   * dplyr   -> verbs like filter(), group_by(), summarise() for tables.
#   * ggplot2 -> the plotting system (grammar of graphics).
# Both live inside the "tidyverse" bundle.
#
# install.packages() DOWNLOADS a package (do this ONCE per machine). library()
# LOADS it into the current session (do this EVERY time you start R).
# The `if (!requireNamespace(...))` guard means "only install if missing", so
# you can safely leave these lines in the script.
if (!requireNamespace("dplyr",   quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("readr",   quietly = TRUE)) install.packages("readr")

library(dplyr)     # data manipulation
library(ggplot2)   # plotting
library(readr)     # fast, friendly CSV reading


# -----------------------------------------------------------------------------
# STEP 1 — CONFIGURATION (the equivalent of plot_config.py)
# -----------------------------------------------------------------------------
# We gather all the "magic values" (colours, labels, file paths) in one place.
# In R we can store a lookup table as a *named vector* or a *named list*:
# `GROUP_COLORS[["HC"]]` returns "steelblue". This mirrors the Python dicts.

# Colour for each group's line. Names on the left, colours on the right.
GROUP_COLORS <- c(HC = "steelblue", T = "firebrick")

# Human-readable y-axis label for each metric. Add more as needed — the keys
# must match the text in the CSV's `Metric` column exactly.
METRIC_LABELS <- c(
  RMSSD    = "RMSSD (ms)",
  SDNN     = "SDNN (ms)",
  mean_HR  = "Mean HR (bpm)",
  mean_RRI = "Mean RRI (ms)",
  LF       = "LF power (ms^2)",
  HF       = "HF power (ms^2)"
)

# Suffix appended to the y label depending on which "Value_type" we plot.
VALUE_TYPE_LABELS <- c(
  raw        = "",
  diff       = "delta from baseline",
  pct_change = "% change from baseline",
  log_ratio  = "log-ratio vs baseline"
)

# Transparency of the shaded error band (0 = invisible, 1 = solid).
ERROR_ALPHA <- 0.3
# Default confidence level, used only when error_band = "ci".
CI_LEVEL <- 0.95

# Where the input CSVs live and where figures should be written. `file.path()`
# joins folders with the correct separator for your operating system.
# NOTE: adjust PROJECT_ROOT to wherever the repository sits on your machine.
PROJECT_ROOT <- ".."                                   # one level above /R
RESULTS_PATH <- file.path(PROJECT_ROOT, "Results")
FIGURES_PATH <- file.path(PROJECT_ROOT, "Plots", "R_figures")


# -----------------------------------------------------------------------------
# STEP 2 — IMPORT: read the CSV into a data frame
# -----------------------------------------------------------------------------
# A "data frame" is R's spreadsheet: rows = observations, columns = variables.
# read_csv() turns the file on disk into that in-memory table.
#
# The results file is in "long format": each row is ONE measurement of ONE
# metric for ONE participant in ONE 30-second interval. The columns we care
# about are:
#   participant, groupe, condition, task_moment, recording_type,
#   Metric, Value_type, Value,
#   time_interval_rel_start, time_interval_rel_end
load_results <- function(csv_path) {
  # show_col_types = FALSE just silences a chatty message.
  df <- readr::read_csv(csv_path, show_col_types = FALSE)
  message("Loaded ", nrow(df), " rows from ", csv_path)
  return(df)
}


# -----------------------------------------------------------------------------
# STEP 3 — FILTER: keep only the rows we want to plot
#          (this is the R version of `filter_interval_data`)
# -----------------------------------------------------------------------------
# Out of the whole file we want, for a single chart:
#   * only the 30-second "interval" rows (recording_type == "interval"), which
#     drops the per-trial "total" summaries and the interval-less baseline;
#   * only ONE metric (e.g. "RMSSD");
#   * only ONE value_type (e.g. "raw" or "pct_change").
#
# We also:
#   * keep only successfully-computed rows (status == "SUCCESS"),
#   * convert Value and the time columns to numbers (as.numeric),
#   * ROUND the interval start time. Why round? Two participants that both start
#     an interval at "300 seconds" might be stored as 299.999 and 300.0001 due
#     to floating-point noise. Rounding snaps them to the same 300, so later
#     they average into ONE point on the x-axis instead of two nearly-identical
#     ones.
#   * drop rows with no Value or no start time (they cannot be plotted).
#
# In dplyr, the pipe operator `%>%` reads as "and then": take df, THEN filter,
# THEN mutate, and so on. Each verb hands its result to the next.
filter_interval_data <- function(df, metric, value_type,
                                 require_success = TRUE, round_decimals = 0) {
  d <- df %>%
    filter(recording_type == "interval",   # keep only interval rows
           Metric == metric,                # keep only this metric
           Value_type == value_type)        # keep only this value_type

  # Only apply the status filter if that column actually exists in the file.
  if (require_success && "status" %in% names(d)) {
    d <- d %>% filter(status == "SUCCESS")
  }

  d <- d %>%
    mutate(
      Value                   = as.numeric(Value),
      time_interval_rel_start = round(as.numeric(time_interval_rel_start),
                                      round_decimals),
      time_interval_rel_end   = round(as.numeric(time_interval_rel_end),
                                      round_decimals)
    ) %>%
    # Remove rows where the conversion produced NA (missing) values.
    filter(!is.na(Value), !is.na(time_interval_rel_start))

  return(d)
}


# -----------------------------------------------------------------------------
# STEP 4 — AGGREGATE: mean across participants + an error band
#          (this is the R version of `compute_mean_and_band`)
# -----------------------------------------------------------------------------
# After filtering, we may have many rows sharing the same start time (one per
# participant). We collapse them into a single summary row per time point:
#   * mean  = average metric value at that time,
#   * n     = how many participants contributed,
#   * a band showing uncertainty of that mean.
#
# The band can be:
#   * "sem" -> mean +/- 1 standard error of the mean. SEM = sd / sqrt(n).
#   * "ci"  -> a t-distribution confidence interval, mean +/- t* * SEM, where
#              t* comes from qt() (R's t quantile function) for n-1 degrees of
#              freedom.
#
# A band needs at least 2 participants (you cannot estimate spread from one
# value), so where n < 2 we set lower/upper to NA and simply won't shade there.
compute_mean_and_band <- function(d, error_band = "sem", ci_level = CI_LEVEL) {

  # group_by + summarise = "for each start time, compute these numbers".
  agg <- d %>%
    group_by(x = time_interval_rel_start) %>%       # rename the key to `x`
    summarise(
      mean = mean(Value),
      sd   = sd(Value),      # sample sd (divides by n-1), like pandas .std()
      n    = dplyr::n(),     # number of rows (participants) in this bin
      .groups = "drop"       # ungroup afterwards, keeps dplyr quiet
    ) %>%
    arrange(x)               # sort left-to-right along the time axis

  # Standard error of the mean for each time point.
  agg <- agg %>% mutate(sem = sd / sqrt(n))

  # Build the half-width ("margin") of the band, depending on the method.
  if (error_band == "sem") {
    agg <- agg %>% mutate(margin = sem)
  } else if (error_band == "ci") {
    # qt() gives the t critical value; two-sided => use 0.5 + ci_level/2.
    agg <- agg %>%
      mutate(margin = qt(0.5 + ci_level / 2, df = n - 1) * sem)
  } else {
    stop("error_band must be 'sem' or 'ci', got: ", error_band)
  }

  # Where n < 2 the band is undefined -> set margin to NA so nothing is shaded.
  agg <- agg %>%
    mutate(
      margin = ifelse(n >= 2, margin, NA_real_),
      lower  = mean - margin,
      upper  = mean + margin
    )

  return(agg)   # columns: x, mean, sd, n, sem, margin, lower, upper
}


# -----------------------------------------------------------------------------
# STEP 5 — EVENT STRUCTURE: where does "task" end and "recovery" begin?
#          (this is the R version of `derive_event_segments`)
# -----------------------------------------------------------------------------
# Each interval row is tagged with a `task_moment` ("task" or "recovery"). We
# want to draw a vertical line at each boundary and a label over each span,
# WITHOUT hardcoding where those boundaries are — we read them from the data.
#
# Logic: sort intervals by start time, then walk through them merging
# neighbours that share the same task_moment into one continuous "segment".
# We return a small table: one row per segment with its start, end and label.
derive_event_segments <- function(dc) {
  # First, one row per unique start time with its end + task_moment + condition.
  intervals <- dc %>%
    group_by(time_interval_rel_start) %>%
    summarise(
      end         = max(time_interval_rel_end),
      task_moment = first(task_moment),
      condition   = first(condition),
      .groups = "drop"
    ) %>%
    arrange(time_interval_rel_start)

  # Now collapse consecutive rows that share a task_moment. `rle()` (run-length
  # encoding) finds runs of identical values; we use its group ids to summarise.
  runs <- rle(intervals$task_moment)
  intervals$run_id <- rep(seq_along(runs$lengths), runs$lengths)

  segments <- intervals %>%
    group_by(run_id) %>%
    summarise(
      start       = min(time_interval_rel_start),
      end         = max(end),
      task_moment = first(task_moment),
      condition   = first(condition),
      .groups = "drop"
    ) %>%
    # Label a "task" span with the condition name; otherwise use the moment.
    mutate(label = ifelse(task_moment == "task", condition, task_moment)) %>%
    arrange(start)

  return(segments)
}


# -----------------------------------------------------------------------------
# STEP 6 — PLOT ONE CONDITION
# -----------------------------------------------------------------------------
# This builds the actual ggplot figure for a SINGLE condition, overlaying every
# group. ggplot works by ADDING layers with `+`:
#   ggplot(data) + geom_ribbon(...) + geom_line(...) + labels + theme
# Draw order matters: we add the shaded band FIRST so the lines sit on top.
#
# `plot_df`     -> the aggregated table for all groups, with a `group` column.
# `segments`    -> the event table from derive_event_segments().
plot_one_condition <- function(plot_df, segments, metric, value_type,
                               condition, error_band) {

  # --- y-axis label: metric label + value_type suffix -----------------------
  base_lbl <- ifelse(metric %in% names(METRIC_LABELS),
                     METRIC_LABELS[[metric]], metric)
  suffix   <- ifelse(value_type %in% names(VALUE_TYPE_LABELS),
                     VALUE_TYPE_LABELS[[value_type]], value_type)
  y_lab    <- if (suffix == "") base_lbl else paste0(base_lbl, " — ", suffix)
  band_lbl <- if (error_band == "sem") "+/-1 SEM"
              else paste0(round(CI_LEVEL * 100), "% CI")

  # --- data-driven x tick positions: each interval start + the final end ----
  xticks <- sort(unique(plot_df$x))
  xticks <- c(xticks, max(segments$end))

  # --- vertical y position for the segment labels ---------------------------
  # Put the text near the top of the data range.
  y_top    <- max(plot_df$upper, plot_df$mean, na.rm = TRUE)
  y_bottom <- min(plot_df$lower, plot_df$mean, na.rm = TRUE)
  y_text   <- y_top - (y_top - y_bottom) * 0.03
  # Boundary x positions = first start + every segment end.
  boundaries <- c(segments$start[1], segments$end)
  segments$x_center <- (segments$start + segments$end) / 2

  # --- assemble the plot ----------------------------------------------------
  p <- ggplot(plot_df, aes(x = x, y = mean, colour = group, fill = group)) +

    # 1) shaded error band. `data = subset(...)` drops NA band rows so
    #    geom_ribbon does not warn. colour = NA => no outline on the band.
    geom_ribbon(
      data = subset(plot_df, !is.na(lower) & !is.na(upper)),
      aes(ymin = lower, ymax = upper),
      alpha = ERROR_ALPHA, colour = NA
    ) +

    # 2) the mean line with a marker at every time point.
    geom_line(linewidth = 1) +
    geom_point(size = 1.6) +

    # 3) dashed vertical lines at each task/recovery boundary.
    geom_vline(xintercept = boundaries, linetype = "dashed",
               colour = "grey45", linewidth = 0.5) +

    # 4) a label centred over each segment (task condition / "recovery").
    geom_label(
      data = segments,
      aes(x = x_center, y = y_text, label = label),
      inherit.aes = FALSE,               # don't reuse the group colour mapping
      size = 3, colour = "grey20", linewidth = 0.2, alpha = 0.75
    ) +

    # 5) map our colour names to the actual colours defined in GROUP_COLORS.
    scale_colour_manual(values = GROUP_COLORS, name = "Group") +
    scale_fill_manual(values = GROUP_COLORS, guide = "none") +

    # 6) put the ticks exactly where the intervals start.
    scale_x_continuous(breaks = xticks) +

    # 7) titles, axis labels, and a clean theme.
    labs(
      title = paste0(metric, " — ", condition, "  (band: ", band_lbl, ")"),
      x = "Within-trial time (s) — interval start",
      y = y_lab
    ) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  return(p)
}


# -----------------------------------------------------------------------------
# STEP 7 — THE MAIN ENTRY POINT (the R version of `plot_all_task_all_group`)
# -----------------------------------------------------------------------------
# Tie everything together: filter -> for each condition, for each group
# aggregate -> plot -> save. Returns a named list of ggplot objects so you can
# also inspect them interactively at the R console.
plot_all_task_all_group <- function(df, metric, value_type = "raw",
                                    error_band = "sem",
                                    conditions = NULL, groups = NULL,
                                    save = TRUE,
                                    figures_path = FIGURES_PATH) {

  # 1) FILTER down to this metric / value_type.
  d <- filter_interval_data(df, metric, value_type)
  if (nrow(d) == 0) {
    message("No interval rows for metric=", metric,
            ", value_type=", value_type, ". Nothing to plot.")
    return(invisible(list()))
  }

  # 2) Decide which conditions and groups to draw if the caller didn't say.
  #    Default conditions = every non-baseline condition present.
  if (is.null(conditions)) {
    conditions <- unique(d$condition)
    conditions <- conditions[conditions != "baseline"]
  }
  if (is.null(groups)) {
    groups <- sort(unique(d$groupe[!is.na(d$groupe)]))
  }

  # Make sure the output folder exists before we try to save into it.
  if (save) dir.create(figures_path, recursive = TRUE, showWarnings = FALSE)

  figures <- list()   # we will collect one plot per condition here.

  # 3) Loop over conditions — one figure each.
  for (condition in conditions) {
    dc <- d %>% filter(condition == !!condition)   # !! = "use the variable"
    if (nrow(dc) == 0) {
      message("  (skip) condition ", condition, ": no data")
      next
    }

    # 3a) For each group, aggregate, then stack the results into one table
    #     with a `group` column so ggplot can colour by it.
    per_group <- list()
    for (g in groups) {
      dg <- dc %>% filter(groupe == g)
      if (nrow(dg) == 0) next
      agg <- compute_mean_and_band(dg, error_band = error_band)
      agg$group <- g                       # tag every row with its group
      agg$n_part <- length(unique(dg$participant))
      per_group[[g]] <- agg
    }
    if (length(per_group) == 0) next
    plot_df <- bind_rows(per_group)         # one long table for all groups

    # 3b) Reconstruct the task/recovery event structure for this condition.
    segments <- derive_event_segments(dc)

    # 3c) Build the figure.
    p <- plot_one_condition(plot_df, segments, metric, value_type,
                            condition, error_band)
    figures[[condition]] <- p

    # 3d) Save to disk. ggsave() writes whatever plot you hand it.
    if (save) {
      fname <- file.path(
        figures_path,
        paste0("timeseries_", metric, "_", value_type, "_",
               error_band, "_", condition, ".png")
      )
      ggsave(fname, plot = p, width = 12, height = 4, dpi = 150)
      message("  Saved: ", fname)
    }
  }

  return(invisible(figures))
}


# =============================================================================
#  RUN IT — example usage
# =============================================================================
# Everything above only DEFINES functions; nothing has run yet. The block below
# actually executes the pipeline. When sourcing this whole file you may want to
# comment this out; when learning, run these lines one at a time in the console
# and inspect the intermediate objects (they are ordinary data frames).
#
# To run:  from an R session in the /R folder, type  source("plot_hrv_timeseries.R")
#          or step through interactively.

if (interactive()) {

  # --- 1. IMPORT --------------------------------------------------------------
  temp_csv <- file.path(RESULTS_PATH, "processed_ppg_results_temp.csv")
  df <- load_results(temp_csv)

  # --- Peek at the intermediate tables (great for learning) -------------------
  # d   <- filter_interval_data(df, "RMSSD", "raw")
  # head(d)                          # the filtered rows
  # agg <- compute_mean_and_band(d %>% filter(groupe == "HC"))
  # head(agg)                        # mean + band per time point
  # derive_event_segments(d %>% filter(condition == unique(d$condition)[2]))

  # --- 2. FULL PIPELINE -> one figure per condition ---------------------------
  figs <- plot_all_task_all_group(
    df,
    metric     = "RMSSD",
    value_type = "raw",
    error_band = "sem",   # or "ci" for a 95% confidence interval band
    save       = TRUE
  )

  # `figs` is a list keyed by condition. To view one on screen:
  #   print(figs[[1]])
}
