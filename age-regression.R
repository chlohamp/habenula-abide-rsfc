library("lmerTest")
library("readr")
library("dplyr")
library("tidyr")
library("ggplot2")

# Fix font/graphics issues on macOS
options(bitmapType = 'cairo')
if (capabilities("cairo")) {
  options(device = function(...) cairo_pdf(...))
}

# Set paths
data_dir <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/group-drawn/age-effect5-21/age-differences/"
output_dir <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/group-drawn/age-effect5-21/age-differences/"
participants_file <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/participants.tsv"

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load participants data for Sex and Site
participants_df <- read_tsv(participants_file, show_col_types = FALSE) %>%
  select(participant_id, SEX, SITE_ID) %>%
  rename(Subject = participant_id, Sex = SEX, Site = SITE_ID)

# Load cluster data files
clusters <- c("1")
cluster_data_list <- list()

for (cluster in clusters) {
  csv_file <- paste0(data_dir, "age-connectivity-cluster", cluster, ".csv")
  if (file.exists(csv_file)) {
    cluster_df <- read_csv(csv_file, show_col_types = FALSE)
    # Rename Mean_Zscore to Correlation
    cluster_df <- cluster_df %>% rename(Correlation = Mean_Zscore)
    # Merge with participants data to get Sex and Site
    cluster_df <- cluster_df %>% left_join(participants_df, by = "Subject")
    cluster_data_list[[cluster]] <- cluster_df
    message(paste("Loaded:", csv_file))
  } else {
    warning(paste("File not found:", csv_file))
  }
}

# Combine all cluster data
combined_data <- bind_rows(cluster_data_list)

# Calculate overall sample sizes by group
message("\n=== Overall Sample Sizes (Unique Subjects) ===")
sample_sizes_overall <- combined_data %>%
  distinct(Subject, Group) %>%
  group_by(Group) %>%
  summarise(N = n(), .groups = 'drop')

n_total <- combined_data %>% distinct(Subject) %>% nrow()
n_asd <- ifelse("asd" %in% sample_sizes_overall$Group, 
                sample_sizes_overall$N[sample_sizes_overall$Group == "asd"], 0)
n_td <- ifelse("td" %in% sample_sizes_overall$Group, 
               sample_sizes_overall$N[sample_sizes_overall$Group == "td"], 0)

message(sprintf("Total: %d subjects (ASD: %d, TD: %d)", n_total, n_asd, n_td))

# Save overall sample sizes
overall_sample_sizes <- data.frame(
  Total_N = n_total,
  ASD_N = n_asd,
  TD_N = n_td
)
write_csv(overall_sample_sizes, paste0(output_dir, "overall_sample_sizes.csv"))
message(paste("Saved overall sample sizes to:", paste0(output_dir, "overall_sample_sizes.csv")))
message("==============================================\n")

# Save cluster-specific CSV files for statistical analysis
for (cluster in clusters) {
  if (!is.null(cluster_data_list[[cluster]])) {
    cluster_df <- cluster_data_list[[cluster]]
    output_file <- paste0(output_dir, "cluster-", cluster, "_age_data.csv")
    write_csv(cluster_df, output_file)
    message(paste("Saved:", output_file))
  }
}

# Run statistical models for age effects
roi <- "Correlation"  # Habenula connectivity
group_var <- "Group"
categorical_vars <- c("Sex")
numerical_vars <- c("Age")

# Store p-values for multiple comparison correction
all_pvalues <- list()

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_age_data_weighted.csv")
  
  if (!file.exists(data_path)) {
    message(paste("Skipping cluster", cluster, "- no data file"))
    next
  }
  
  data <- read_csv(data_path, show_col_types = FALSE)
  
  message(paste("\n=== Processing Cluster", cluster, "==="))
  
  # Prepare data for modeling
  all_columns <- c(roi, categorical_vars, numerical_vars, group_var, "Site")
  sub_data <- data[, all_columns]
  sub_data <- na.omit(sub_data)
  
  # Skip if insufficient data
  if (nrow(sub_data) < 10) {
    message(paste("Skipping cluster", cluster, "- insufficient data"))
    next
  }
  
  # Convert categorical variables to factors
  for (var in categorical_vars) {
    sub_data[[var]] <- factor(sub_data[[var]])
  }
  
  # Relevel Group to make asd the reference
  sub_data[[group_var]] <- relevel(factor(sub_data[[group_var]]), ref = "asd")
  
  # Ensure numeric columns are numeric
  sub_data[[roi]] <- as.numeric(sub_data[[roi]])
  for (var in numerical_vars) {
    sub_data[[var]] <- as.numeric(sub_data[[var]])
  }
  
  # Extract age slopes for each group separately
  tryCatch({
    # Split data by group
    asd_data <- sub_data %>% filter(Group == "asd")
    td_data <- sub_data %>% filter(Group == "td")
    
    # Fit simple linear models for each group (Age predicting Connectivity)
    asd_model <- lm(as.formula(paste(roi, "~ Age")), data = asd_data)
    td_model <- lm(as.formula(paste(roi, "~ Age")), data = td_data)
    
    # Extract slope coefficients
    asd_slope <- coef(summary(asd_model))["Age", "Estimate"]
    asd_se <- coef(summary(asd_model))["Age", "Std. Error"]
    asd_pval <- coef(summary(asd_model))["Age", "Pr(>|t|)"]
    asd_ci_lower <- asd_slope - 1.96 * asd_se
    asd_ci_upper <- asd_slope + 1.96 * asd_se
    
    td_slope <- coef(summary(td_model))["Age", "Estimate"]
    td_se <- coef(summary(td_model))["Age", "Std. Error"]
    td_pval <- coef(summary(td_model))["Age", "Pr(>|t|)"]
    td_ci_lower <- td_slope - 1.96 * td_se
    td_ci_upper <- td_slope + 1.96 * td_se
    
    # Calculate difference in slopes
    slope_diff <- asd_slope - td_slope
    
    # Test the interaction using full model with Group*Age interaction
    full_model <- lm(as.formula(paste(roi, "~ Group * Age + Sex")), data = sub_data)
    interaction_pval <- coef(summary(full_model))["Grouptd:Age", "Pr(>|t|)"]
    
    # Store p-values for correction
    all_pvalues[[cluster]] <- list(
      asd = asd_pval,
      td = td_pval,
      interaction = interaction_pval
    )
    
    # Calculate Cohen's d for the slope difference
    # Use the pooled SD of the outcome variable (connectivity)
    pooled_sd <- sd(sub_data[[roi]], na.rm = TRUE)
    
    # Cohen's d = slope difference × age_range / pooled_sd
    age_range <- max(sub_data$Age, na.rm = TRUE) - min(sub_data$Age, na.rm = TRUE)
    cohens_d <- (slope_diff * age_range) / pooled_sd
    
    # Calculate R-squared for each group to show variance explained
    asd_rsq <- summary(asd_model)$r.squared
    td_rsq <- summary(td_model)$r.squared
    
    # Create summary table
    slope_summary <- data.frame(
      Cluster = cluster,
      Group = c("ASD", "TD", "Difference"),
      N = c(nrow(asd_data), nrow(td_data), NA),
      Age_Slope = c(asd_slope, td_slope, slope_diff),
      Std_Error = c(asd_se, td_se, NA),
      CI_Lower = c(asd_ci_lower, td_ci_lower, NA),
      CI_Upper = c(asd_ci_upper, td_ci_upper, NA),
      R_Squared = c(asd_rsq, td_rsq, NA),
      Cohens_d = c(NA, NA, cohens_d)
    )
    
    message("\n=== Age Slopes ===")
    print(slope_summary)
    
    # Save slope summary
    out_file <- paste0(output_dir, "cluster-", cluster, "_age_weighted_slopes.csv")
    write_csv(slope_summary, out_file)
    message(paste("Saved:", out_file))
    
    # Create plot with both groups
    plot_file <- paste0(output_dir, "cluster-", cluster, "_age_weighted_slopes_plot.png")
    
    # Prepare data for plotting
    plot_data <- data.frame(
      RSFC = sub_data[[roi]],
      Age = sub_data[["Age"]],
      Group = sub_data[[group_var]]
    )
    
    # Count subjects by group
    n_asd_cluster <- nrow(asd_data)
    n_td_cluster <- nrow(td_data)
    n_total_cluster <- nrow(plot_data)
    
    # Create subtitle with slope information and effect size, using line breaks and similar style to phenotypic regression
    subtitle_text <- sprintf(
      "N = %d (ASD: %d, TD: %d)\nSlope ASD: %.3f\nSlope TD: %.3f\nDiff: %.3f | Cohen's d: %.3f",
      n_total_cluster, n_asd_cluster, n_td_cluster, asd_slope, td_slope, slope_diff, cohens_d
    )
    # Create scatter plot with regression lines by group, matching style to phenotypic regression
    # Use tryCatch to handle font/graphics issues
    plot_success <- tryCatch({
      p <- ggplot(plot_data, aes(x = Age, y = RSFC, color = Group, fill = Group, linetype = Group, shape = Group)) +
        geom_point(alpha = 0.6, size = 1.8) +
        geom_smooth(method = "lm", se = TRUE, formula = y ~ x) +
        scale_color_manual(values = c("asd" = "#b6d191", "td" = "#87B2EA")) +
        scale_fill_manual(values = c("asd" = "#b6d191", "td" = "#87B2EA")) +
        scale_linetype_manual(values = c("asd" = "solid", "td" = "dashed")) +
        scale_shape_manual(values = c("asd" = 16, "td" = 4)) +
        coord_cartesian(ylim = c(-0.25, 0.25)) +
        labs(
          title = paste("Cluster", cluster, "- Age-Connectivity Slopes by Group"),
          subtitle = subtitle_text,
          x = "Age (years)",
          y = "Habenula Connectivity"
        ) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, family = ""),
          plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30", family = ""),
          legend.key.width = unit(1, "cm"),
          panel.grid.minor = element_blank(),
          text = element_text(family = "")
        )
      
      # Try saving with different devices if one fails
      saved <- FALSE
      
      # Try PNG with Cairo first
      if (!saved) {
        tryCatch({
          png(plot_file, width = 6*300, height = 5*300, res = 300, type = "cairo")
          print(p)
          dev.off()
          saved <- TRUE
          message(paste("Saved plot (Cairo PNG):", plot_file))
        }, error = function(e) {
          if (dev.cur() != 1) dev.off()  # Clean up if device is still open
        })
      }
      
      # Fallback to regular ggsave
      if (!saved) {
        tryCatch({
          ggsave(plot_file, p, width = 6, height = 5, dpi = 300, device = "png")
          saved <- TRUE
          message(paste("Saved plot (ggsave):", plot_file))
        }, error = function(e) {
          message(paste("ggsave failed:", e$message))
        })
      }
      
      if (!saved) {
        message(paste("Warning: Could not save plot for cluster", cluster))
      }
      
      TRUE
    }, error = function(e) {
      message(paste("Plot creation failed for cluster", cluster, ":", e$message))
      FALSE
    })
    
  }, error = function(e) {
    message(paste("Error in cluster", cluster, ":", e$message))
  })
}

message("\nSlope extraction complete! All clusters analyzed with ASD and TD slopes on same graphs.")
