#!Rscript
#!/usr/bin/env Rscript
#
# evaluate_results.R
# Summarise ST-GCN and simple LSTM metrics and compare by forecast horizon.
#

# ─── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)

# ─── Constants ────────────────────────────────────────────────────────────────
STGCN_FILE <- "evaluation_results.csv"
LSTM_FILE  <- "evaluation_simple_lstm.csv"

OUTPUT_STGCN_HORIZON       <- "summary_stgcn_by_horizon.csv"
OUTPUT_STGCN_HORIZON_SCALER <- "summary_stgcn_by_horizon_scaler.csv"
OUTPUT_LSTM_HORIZON        <- "summary_lstm_by_horizon.csv"
OUTPUT_COMBINED_HORIZON    <- "summary_combined_by_horizon.csv"

# ─── Helper: generic summarisation ────────────────────────────────────────────
summarise_metrics <- function(df, group_vars, metrics) {
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

# ─── Load data ────────────────────────────────────────────────────────────────
stgcn_df <- read_csv(STGCN_FILE, col_types = cols())
lstm_df  <- read_csv(LSTM_FILE,  col_types = cols())

# Add a model label
stgcn_df <- stgcn_df %>% mutate(model = "STGCN")
lstm_df  <- lstm_df  %>% mutate(model = "LSTM")

# ─── Summaries ────────────────────────────────────────────────────────────────
# 1) ST-GCN by horizon
stgcn_horizon <- summarise_metrics(
  stgcn_df,
  group_vars = c("horizon", "model"),
  metrics    = c("mean_mse", "mean_rmse", "mean_mae", "mean_mre")
)

# 2) ST-GCN by horizon & adjacency scaler
stgcn_horizon_scaler <- summarise_metrics(
  stgcn_df,
  group_vars = c("horizon", "adj_scaler", "model"),
  metrics    = c("mean_mse", "mean_rmse", "mean_mae", "mean_mre")
)

# 3) LSTM by horizon
lstm_horizon <- summarise_metrics(
  lstm_df,
  group_vars = c("horizon", "model"),
  metrics    = c("mean_mse", "mean_rmse", "mean_mae", "mean_mre")
)

# 4) Combined comparison by horizon
combined_horizon <- bind_rows(stgcn_horizon, lstm_horizon) %>%
  arrange(horizon, model)

# ─── Print to console ─────────────────────────────────────────────────────────
cat("\n=== ST-GCN: metrics by horizon ===\n")
print(stgcn_horizon)

cat("\n=== ST-GCN: metrics by horizon × scaler ===\n")
print(stgcn_horizon_scaler)

cat("\n=== LSTM: metrics by horizon ===\n")
print(lstm_horizon)

cat("\n=== Combined: ST-GCN vs LSTM by horizon ===\n")
print(combined_horizon)

# ─── Save CSVs ────────────────────────────────────────────────────────────────
write_csv(stgcn_horizon,         OUTPUT_STGCN_HORIZON)
write_csv(stgcn_horizon_scaler,  OUTPUT_STGCN_HORIZON_SCALER)
write_csv(lstm_horizon,          OUTPUT_LSTM_HORIZON)
write_csv(combined_horizon,      OUTPUT_COMBINED_HORIZON)