#!/usr/bin/env Rscript
# Dice Similarity Analysis for Habenula Group Maps
# R Script Version

cat("=== Dice Similarity Analysis ===\n\n")

# Load Required Libraries
cat("Loading required libraries...\n")
library(oro.nifti)
library(neurobase)
library(dplyr)
library(ggplot2)
library(gridExtra)

# Define Dice Coefficient Functions

dice_coefficient <- function(mask1, mask2, brain_mask = NULL) {
  # Flatten arrays if needed
  mask1_flat <- as.vector(mask1)
  mask2_flat <- as.vector(mask2)
  
  # Apply brain mask if provided
  if (!is.null(brain_mask)) {
    brain_mask_flat <- as.logical(as.vector(brain_mask))
    mask1_flat <- mask1_flat[brain_mask_flat]
    mask2_flat <- mask2_flat[brain_mask_flat]
  }
  
  # Calculate intersection and sizes
  intersection <- sum(mask1_flat * mask2_flat)
  size1 <- sum(mask1_flat)
  size2 <- sum(mask2_flat)
  
  # Avoid division by zero
  if (size1 + size2 == 0) {
    return(0.0)
  }
  
  dice <- (2.0 * intersection) / (size1 + size2)
  return(dice)
}

weighted_dice_coefficient <- function(map1, map2, brain_mask = NULL) {
  # Flatten arrays if needed
  map1_flat <- as.vector(map1)
  map2_flat <- as.vector(map2)
  
  # Apply brain mask if provided
  if (!is.null(brain_mask)) {
    brain_mask_flat <- as.logical(as.vector(brain_mask))
    map1_flat <- map1_flat[brain_mask_flat]
    map2_flat <- map2_flat[brain_mask_flat]
  }
  
  # Use absolute values to handle negative activations
  map1_abs <- abs(map1_flat)
  map2_abs <- abs(map2_flat)
  
  # Calculate weighted intersection (minimum activation at each voxel)
  weighted_intersection <- sum(pmin(map1_abs, map2_abs))
  
  # Calculate sum of activations
  sum1 <- sum(map1_abs)
  sum2 <- sum(map2_abs)
  
  # Avoid division by zero
  if (sum1 + sum2 == 0) {
    return(0.0)
  }
  
  weighted_dice <- (2.0 * weighted_intersection) / (sum1 + sum2)
  return(weighted_dice)
}

permutation_test_dice <- function(map1, map2, dice_func, n_permutations = 10000, seed = 42) {
  set.seed(seed)
  
  # Ensure arrays are vectors
  map1 <- as.vector(map1)
  map2 <- as.vector(map2)
  
  # Calculate observed Dice
  observed_dice <- dice_func(map1, map2)
  
  # Initialize null distribution
  null_dice <- numeric(n_permutations)
  
  cat(sprintf("Running permutation test with %d permutations...\n", n_permutations))
  
  # Perform permutations
  for (i in 1:n_permutations) {
    if (i %% 1000 == 0) {
      cat(sprintf("  Progress: %d/%d\n", i, n_permutations))
    }
    # Randomly shuffle map2 to break spatial correspondence
    map2_permuted <- sample(map2)
    # Calculate Dice for permuted data
    null_dice[i] <- dice_func(map1, map2_permuted)
  }
  
  # Calculate one-tailed p-value
  p_value <- sum(null_dice >= observed_dice) / n_permutations
  
  return(list(
    p_value = p_value,
    null_distribution = null_dice,
    observed_dice = observed_dice
  ))
}

# Setup Directories and File Paths
cat("\nSetting up file paths...\n")
data_dir <- "./dset"

# Define file paths
group_drawn_dir <- file.path(data_dir, "group-drawn/habenula")
group_avg_dir <- file.path(data_dir, "group-avg/habenula")

# File paths for group average (1-sample) maps
drawn_1s_fn <- file.path(group_drawn_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz")
avg_1s_fn <- file.path(group_avg_dir, "averaged", "sub-group_task-rest_desc-1SampletTest_thresh.nii.gz")

# File paths for group comparison (2-sample) maps
drawn_2s_fn <- file.path(group_drawn_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz")
avg_2s_fn <- file.path(group_avg_dir, "difference", "sub-group_task-rest_desc-2SampletTest_thresh.nii.gz")

cat("Drawn 1-sample:", drawn_1s_fn, "\n")
cat("Avg 1-sample:", avg_1s_fn, "\n")
cat("Drawn 2-sample:", drawn_2s_fn, "\n")
cat("Avg 2-sample:", avg_2s_fn, "\n")

# Calculate Dice Coefficients for Group Average Maps (1-Sample)
cat("\n=== Group Average (1-Sample) Analysis ===\n")

if (!file.exists(drawn_1s_fn)) {
  cat("ERROR: File not found:", drawn_1s_fn, "\n")
} else if (!file.exists(avg_1s_fn)) {
  cat("ERROR: File not found:", avg_1s_fn, "\n")
} else {
  cat("Loading 1-sample maps...\n")
  # Load thresholded maps
  drawn_1s_img <- readNIfTI(drawn_1s_fn, reorient = FALSE)
  avg_1s_img <- readNIfTI(avg_1s_fn, reorient = FALSE)
  
  # Extract data arrays
  drawn_1s_arr <- as.vector(drawn_1s_img)
  avg_1s_arr <- as.vector(avg_1s_img)
  
  # Create binary masks (any non-zero value = 1)
  drawn_1s_binary <- as.integer(drawn_1s_arr != 0)
  avg_1s_binary <- as.integer(avg_1s_arr != 0)
  
  # Calculate binary Dice coefficient
  dice_1s <- dice_coefficient(drawn_1s_binary, avg_1s_binary)
  cat(sprintf("Binary Dice Coefficient: %.4f\n", dice_1s))
  
  # Calculate weighted Dice coefficient
  weighted_dice_1s <- weighted_dice_coefficient(drawn_1s_arr, avg_1s_arr)
  cat(sprintf("Weighted Dice Coefficient: %.4f\n", weighted_dice_1s))
  
  # Check if binary maps are identical
  if (all(drawn_1s_binary == avg_1s_binary)) {
    cat("Binary maps are identical - permutation test not applicable\n")
    p_binary_1s <- NA
    null_binary_1s <- dice_1s
    obs_binary_1s <- dice_1s
  } else {
    cat("Running binary permutation test...\n")
    perm_result_binary_1s <- permutation_test_dice(
      drawn_1s_binary, avg_1s_binary, dice_coefficient, n_permutations = 10000
    )
    p_binary_1s <- perm_result_binary_1s$p_value
    null_binary_1s <- perm_result_binary_1s$null_distribution
    obs_binary_1s <- perm_result_binary_1s$observed_dice
  }
  
  cat("Running weighted permutation test...\n")
  perm_result_weighted_1s <- permutation_test_dice(
    drawn_1s_arr, avg_1s_arr, weighted_dice_coefficient, n_permutations = 10000
  )
  p_weighted_1s <- perm_result_weighted_1s$p_value
  null_weighted_1s <- perm_result_weighted_1s$null_distribution
  obs_weighted_1s <- perm_result_weighted_1s$observed_dice
  
  # Print results
  if (is.na(p_binary_1s)) {
    cat(sprintf("\nBinary Dice: %.4f (identical maps)\n", dice_1s))
  } else {
    if (p_binary_1s == 0.0) {
      cat(sprintf("\nBinary Dice: %.4f (p < 0.0001)\n", dice_1s))
    } else {
      cat(sprintf("\nBinary Dice: %.4f (p = %.6f)\n", dice_1s, p_binary_1s))
    }
  }
  
  if (p_weighted_1s == 0.0) {
    cat(sprintf("Weighted Dice: %.4f (p < 0.0001)\n", weighted_dice_1s))
  } else {
    cat(sprintf("Weighted Dice: %.4f (p = %.6f)\n", weighted_dice_1s, p_weighted_1s))
  }
  
  cat(sprintf("Hand-drawn voxels: %d\n", sum(drawn_1s_binary)))
  cat(sprintf("Avg habenula voxels: %d\n", sum(avg_1s_binary)))
  cat(sprintf("Overlapping voxels: %d\n", sum(drawn_1s_binary * avg_1s_binary)))
}

# Calculate Dice Coefficients for Group Comparison Maps (2-Sample)
cat("\n=== Group Comparison (2-Sample) Analysis ===\n")

if (!file.exists(drawn_2s_fn)) {
  cat("ERROR: File not found:", drawn_2s_fn, "\n")
} else if (!file.exists(avg_2s_fn)) {
  cat("ERROR: File not found:", avg_2s_fn, "\n")
} else {
  cat("Loading 2-sample maps...\n")
  # Load thresholded maps
  drawn_2s_img <- readNIfTI(drawn_2s_fn, reorient = FALSE)
  avg_2s_img <- readNIfTI(avg_2s_fn, reorient = FALSE)
  
  # Extract data arrays
  drawn_2s_arr <- as.vector(drawn_2s_img)
  avg_2s_arr <- as.vector(avg_2s_img)
  
  # Create binary masks
  drawn_2s_binary <- as.integer(drawn_2s_arr != 0)
  avg_2s_binary <- as.integer(avg_2s_arr != 0)
  
  # Calculate Dice coefficients
  dice_2s <- dice_coefficient(drawn_2s_binary, avg_2s_binary)
  cat(sprintf("Binary Dice Coefficient: %.4f\n", dice_2s))
  
  weighted_dice_2s <- weighted_dice_coefficient(drawn_2s_arr, avg_2s_arr)
  cat(sprintf("Weighted Dice Coefficient: %.4f\n", weighted_dice_2s))
  
  # Permutation tests
  cat("Running binary permutation test...\n")
  perm_result_binary_2s <- permutation_test_dice(
    drawn_2s_binary, avg_2s_binary, dice_coefficient, n_permutations = 10000
  )
  p_binary_2s <- perm_result_binary_2s$p_value
  null_binary_2s <- perm_result_binary_2s$null_distribution
  obs_binary_2s <- perm_result_binary_2s$observed_dice
  
  cat("Running weighted permutation test...\n")
  perm_result_weighted_2s <- permutation_test_dice(
    drawn_2s_arr, avg_2s_arr, weighted_dice_coefficient, n_permutations = 10000
  )
  p_weighted_2s <- perm_result_weighted_2s$p_value
  null_weighted_2s <- perm_result_weighted_2s$null_distribution
  obs_weighted_2s <- perm_result_weighted_2s$observed_dice
  
  # Print results
  if (p_binary_2s == 0.0) {
    cat(sprintf("\nBinary Dice: %.4f (p < 0.0001)\n", dice_2s))
  } else {
    cat(sprintf("\nBinary Dice: %.4f (p = %.6f)\n", dice_2s, p_binary_2s))
  }
  
  if (p_weighted_2s == 0.0) {
    cat(sprintf("Weighted Dice: %.4f (p < 0.0001)\n", weighted_dice_2s))
  } else {
    cat(sprintf("Weighted Dice: %.4f (p = %.6f)\n", weighted_dice_2s, p_weighted_2s))
  }
  
  cat(sprintf("Hand-drawn voxels: %d\n", sum(drawn_2s_binary)))
  cat(sprintf("Avg habenula voxels: %d\n", sum(avg_2s_binary)))
  cat(sprintf("Overlapping voxels: %d\n", sum(drawn_2s_binary * avg_2s_binary)))
}

# Visualize Null Distributions
cat("\n=== Generating Visualizations ===\n")

# Create plots
p1 <- ggplot(data.frame(dice = null_binary_1s), aes(x = dice)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "gray", color = "black") +
  geom_vline(xintercept = obs_binary_1s, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Dice Coefficient", y = "Frequency", 
       title = sprintf("1-Sample Binary (p=%.4f)", ifelse(is.na(p_binary_1s), 0, p_binary_1s))) +
  theme_minimal()

p2 <- ggplot(data.frame(dice = null_weighted_1s), aes(x = dice)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "gray", color = "black") +
  geom_vline(xintercept = obs_weighted_1s, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Dice Coefficient", y = "Frequency", 
       title = sprintf("1-Sample Weighted (p=%.4f)", p_weighted_1s)) +
  theme_minimal()

p3 <- ggplot(data.frame(dice = null_binary_2s), aes(x = dice)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "gray", color = "black") +
  geom_vline(xintercept = obs_binary_2s, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Dice Coefficient", y = "Frequency", 
       title = sprintf("2-Sample Binary (p=%.4f)", p_binary_2s)) +
  theme_minimal()

p4 <- ggplot(data.frame(dice = null_weighted_2s), aes(x = dice)) +
  geom_histogram(bins = 50, alpha = 0.7, fill = "gray", color = "black") +
  geom_vline(xintercept = obs_weighted_2s, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Dice Coefficient", y = "Frequency", 
       title = sprintf("2-Sample Weighted (p=%.4f)", p_weighted_2s)) +
  theme_minimal()

# Save plot
plot_file <- file.path(data_dir, "dice_permutation_distributions_R.png")
ggsave(plot_file, arrangeGrob(p1, p2, p3, p4, ncol = 2),
       width = 12, height = 8, dpi = 300)
cat("Saved plot to:", plot_file, "\n")

# Print null distribution statistics
cat("\nNull distribution statistics:\n")
cat(sprintf("1-Sample Binary: mean=%.4f, observed=%.4f\n", mean(null_binary_1s), obs_binary_1s))
cat(sprintf("1-Sample Weighted: mean=%.4f, observed=%.4f\n", mean(null_weighted_1s), obs_weighted_1s))
cat(sprintf("2-Sample Binary: mean=%.4f, observed=%.4f\n", mean(null_binary_2s), obs_binary_2s))
cat(sprintf("2-Sample Weighted: mean=%.4f, observed=%.4f\n", mean(null_weighted_2s), obs_weighted_2s))

# Summary Table
cat("\n=== Summary Table ===\n")

format_pvalue <- function(p) {
  if (is.na(p)) {
    return("N/A (identical)")
  } else if (p == 0.0) {
    return("< 0.0001")
  } else {
    return(sprintf("%.6f", p))
  }
}

summary_data <- data.frame(
  `Analysis_Type` = c("Group Average (1-Sample)", "Group Comparison (2-Sample)"),
  `Binary_Dice` = c(sprintf("%.4f", dice_1s), sprintf("%.4f", dice_2s)),
  `Binary_p_value` = c(format_pvalue(p_binary_1s), format_pvalue(p_binary_2s)),
  `Weighted_Dice` = c(sprintf("%.4f", weighted_dice_1s), sprintf("%.4f", weighted_dice_2s)),
  `Weighted_p_value` = c(format_pvalue(p_weighted_1s), format_pvalue(p_weighted_2s)),
  `Hand_Drawn_Voxels` = c(sum(drawn_1s_binary), sum(drawn_2s_binary)),
  `Avg_Habenula_Voxels` = c(sum(avg_1s_binary), sum(avg_2s_binary)),
  `Overlapping_Voxels` = c(
    sum(drawn_1s_binary * avg_1s_binary),
    sum(drawn_2s_binary * avg_2s_binary)
  )
)

print(summary_data)

# Save to CSV
output_fn <- file.path(data_dir, "dice_similarity_results_R.csv")
write.csv(summary_data, output_fn, row.names = FALSE)
cat("\nResults saved to:", output_fn, "\n")

# Robustness Check
cat("\n=== Robustness Check ===\n")

check_robustness <- function(observed, null, label) {
  mean_null <- mean(null)
  std_null <- sd(null)
  z_score <- ifelse(std_null > 0, (observed - mean_null) / std_null, NA)
  
  cat(sprintf("\n%s:\n", label))
  cat(sprintf("  Observed Dice: %.4f\n", observed))
  cat(sprintf("  Null mean:     %.4f\n", mean_null))
  cat(sprintf("  Null std:      %.4f\n", std_null))
  cat(sprintf("  Z-score:       %.2f\n", z_score))
  
  if (!is.na(z_score)) {
    if (z_score > 3) {
      cat("  Result is robust: observed value is far outside the null distribution (z > 3)\n")
    } else if (z_score > 2) {
      cat("  Result is likely robust (z > 2)\n")
    } else {
      cat("  Result is not far outside the null (z <= 2)\n")
    }
  }
}

check_robustness(obs_binary_1s, null_binary_1s, "1-Sample Binary Dice")
check_robustness(obs_weighted_1s, null_weighted_1s, "1-Sample Weighted Dice")
check_robustness(obs_binary_2s, null_binary_2s, "2-Sample Binary Dice")
check_robustness(obs_weighted_2s, null_weighted_2s, "2-Sample Weighted Dice")

cat("\n=== Analysis Complete ===\n")
