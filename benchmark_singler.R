# --- 1. Load necessary libraries ---
suppressPackageStartupMessages({
  library(SingleR)
  library(SingleCellExperiment)
  library(readr)
  library(dplyr)
  library(tibble)
  library(scuttle)
  library(MLmetrics)
})

# --- 2. Define input/output paths ---
train_matrix_path <- "benchmark_data/train_set_count_matrix.csv"
train_labels_path <- "benchmark_data/train_set_labels.csv"
test_matrix_path <- "benchmark_data/test_set_count_matrix.csv"
test_labels_path <- "benchmark_data/test_set_labels.csv"

output_dir <- "benchmark_results/singler/"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_predictions_path <- file.path(output_dir, "singleR_predictions.csv")
report_path <- file.path(output_dir, "report_singler.txt")

start_time_total <- Sys.time()

# --- 3. Load and prepare data ---
cat("--- Step 1/4: Loading and preparing data ---\n")
start_time_load <- Sys.time()

tryCatch({
  train_counts_df <- read_csv(train_matrix_path) %>% column_to_rownames("cell_id")
  train_labels <- read_csv(train_labels_path)
  test_counts_df <- read_csv(test_matrix_path) %>% column_to_rownames("cell_id")
  test_labels <- read_csv(test_labels_path)
}, error = function(e) {
  stop(paste("Data loading failed:", e$message))
})

common_genes <- intersect(colnames(train_counts_df), colnames(test_counts_df))
train_counts_df <- train_counts_df[, common_genes]
test_counts_df <- test_counts_df[, common_genes]

train_counts_mat <- t(as.matrix(train_counts_df))
test_counts_mat <- t(as.matrix(test_counts_df))

cat(paste("Data dimension check (genes x cells):\n"))
cat(paste("  - Training set:", dim(train_counts_mat)[1], "genes x", dim(train_counts_mat)[2], "cells\n"))
cat(paste("  - Test set:", dim(test_counts_mat)[1], "genes x", dim(test_counts_mat)[2], "cells\n"))

train_sce <- SingleCellExperiment(assays = list(counts = train_counts_mat))
test_sce <- SingleCellExperiment(assays = list(counts = test_counts_mat))

train_sce <- logNormCounts(train_sce)
test_sce <- logNormCounts(test_sce)
cat("Data has been log-normalized.\n")

train_sce$label <- train_labels$class[match(colnames(train_sce), train_labels$cell_id)]
test_sce$label <- test_labels$class[match(colnames(test_sce), test_labels$cell_id)]

if(any(is.na(train_sce$label))) {
  warning("NA labels found in the training set, removing these cells...")
  train_sce <- train_sce[, !is.na(train_sce$label)]
}

load_time <- difftime(Sys.time(), start_time_load, units = "secs")
cat("Data preparation complete.\n")

# --- 4. Train and predict ---
cat("--- Step 2/4: Training reference and making predictions ---\n")
start_time_run <- Sys.time()
predictions <- SingleR(test = test_sce, ref = train_sce, labels = train_sce$label)
run_time <- difftime(Sys.time(), start_time_run, units = "secs")
cat("SingleR prediction complete.\n")

# --- 5. Save prediction results ---
cat("--- Step 3/4: Saving prediction results to CSV ---\n")
results_df <- data.frame(
  cell_id = colnames(test_sce),
  true_label = test_sce$label,
  predicted_label_singleR = predictions$labels,
  stringsAsFactors = FALSE
)
write_csv(results_df, output_predictions_path)
cat(paste("Prediction results have been saved to:", output_predictions_path, "\n"))

# --- 6. Calculate basic metrics and generate report ---
cat("--- Step 4/4: Calculating basic metrics and generating report ---\n")

valid_indices <- !is.na(results_df$true_label) & !is.na(results_df$predicted_label_singleR)
y_true <- results_df$true_label[valid_indices]
y_pred <- results_df$predicted_label_singleR[valid_indices]

# 1. Calculate F1 score for each class
f1_per_class <- F1_Score(y_true = y_true, y_pred = y_pred)
# 2. Manually calculate macro average (i.e., the average of F1 scores for all classes)
f1_macro <- mean(f1_per_class, na.rm = TRUE)

# Calculate accuracy
accuracy <- Accuracy(y_true = y_true, y_pred = y_pred)

# Generate a simple confusion matrix
class_report <- table(Predicted = y_pred, True = y_true)

total_time <- difftime(Sys.time(), start_time_total, units = "secs")

sink(report_path)
cat("========== SingleR Performance Evaluation Report ==========\n\n")
cat(paste("Evaluated", length(y_true), "cells (NA labels excluded).\n\n"))
cat("--- Core Metrics ---\n")
cat(paste("Accuracy:", round(accuracy, 4), "\n"))
cat(paste("F1-Score (Macro):", round(f1_macro, 4), "\n\n"))
cat("--- Runtimes ---\n")
cat(paste("Data preparation:", round(load_time), "seconds\n"))
cat(paste("Training & Prediction:", round(run_time), "seconds\n"))
cat(paste("Total:", round(total_time), "seconds\n\n"))
cat("--- Simple Confusion Matrix (Predicted vs True) ---\n")
print(class_report)
sink()

cat(paste("Evaluation report has been saved to:", report_path, "\n"))
cat("--- SingleR benchmark complete ---\n")