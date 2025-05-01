#!Rscript
#!/usr/bin/env Rscript
#
# evaluate_results.R
# Summarise and report metrics from evaluation_results.csv
#

# ─── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)  # includes readr, dplyr, ggplot2, etc.

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_FILE <- "evaluation_results.csv"

# ─── Helper Function ──────────────────────────────────────────────────────────
# Read the single evaluation CSV and compute grouped means.
summarise_metrics <- function(
  df,
  group_vars = c("horizon"),
  metrics   = c("mean_mse", "mean_rmse", "mean_mae", "mean_mre")
) {
  df %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      across(
        all_of(metrics),
        ~ mean(.x, na.rm = TRUE),
        .names = "avg_{.col}"
      ),
      .groups = "drop"
    )
}

# ─── Main ────────────────────────────────────────────────────────────────────
# 1) Read in the evaluation results
results <- read_csv(INPUT_FILE, col_types = cols())

# 2) Summary by forecast horizon
summary_by_horizon <- summarise_metrics(
  results,
  group_vars = "horizon"
)

# 3) Summary by horizon and adjacency scaler
summary_by_horizon_scaler <- summarise_metrics(
  results,
  group_vars = c("horizon", "adj_scaler")
)

# 4) Print to console
cat("\n=== Metrics by Forecast Horizon ===\n")
print(summary_by_horizon)

cat("\n=== Metrics by Horizon × Adjacency Scaler ===\n")
print(summary_by_horizon_scaler)

# ─── (Optional) Save summaries as CSVs ────────────────────────────────────────
write_csv(summary_by_horizon,         "summary_by_horizon.csv")
write_csv(summary_by_horizon_scaler,  "summary_by_horizon_scaler.csv")