library("lmerTest")
library("readr")
library("dplyr")
library("tidyr")
library("ggplot2")

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
clusters <- c("1", "2", "3")
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

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_age_data.csv")
  
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
  
  # Build equation with Group*Age interaction
  # Match original 3dLMEr model: group*age+gender+(1|site)
  equation_lmer <- paste(roi, "~ Age * Group + Sex + (1|Site)")
  
  message(paste("Cluster", cluster, "model:", equation_lmer))
  message(paste("Sample size:", nrow(sub_data)))
  
  # Run model with lmer (Site as random effect)
  tryCatch({
    model <- lmer(as.formula(equation_lmer), data = sub_data)
    
    print(summary(model))
    
    # Write results
    out_file <- paste0(output_dir, "cluster-", cluster, "_age_interaction_table.csv")
    model_table <- as.data.frame(coef(summary(model)))
    write.csv(model_table, file = out_file, row.names = TRUE)
    message(paste("Saved:", out_file))
    
    # Create plot
    plot_file <- paste0(output_dir, "cluster-", cluster, "_age_interaction_plot.png")
    
    # Use raw data for plotting with group colors
    plot_data <- data.frame(
      RSFC = sub_data[[roi]],
      Age = sub_data[["Age"]],
      Group = sub_data[[group_var]]
    )
    
    # Count subjects by group
    n_asd_cluster <- sum(plot_data$Group == "asd")
    n_td_cluster <- sum(plot_data$Group == "td")
    n_total_cluster <- nrow(plot_data)
    
    # Create scatter plot with regression lines by group
    p <- ggplot(plot_data, aes(x = Age, y = RSFC, color = Group, fill = Group)) +
      geom_point(alpha = 0.6, size = 2) +
      geom_smooth(method = "lm", se = TRUE) +
      scale_color_manual(values = c("asd" = "#E74C3C", "td" = "#3498DB")) +
      scale_fill_manual(values = c("asd" = "#E74C3C", "td" = "#3498DB")) +
      labs(
        title = paste("Cluster", cluster, "- Age × Group Interaction"),
        subtitle = sprintf("N = %d (ASD: %d, TD: %d)", n_total_cluster, n_asd_cluster, n_td_cluster),
        x = "Age (years)",
        y = "Habenula Connectivity"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30")
      )
    
    ggsave(plot_file, p, width = 7, height = 5, dpi = 300)
    message(paste("Saved plot:", plot_file))
    
  }, error = function(e) {
    message(paste("Error in cluster", cluster, ":", e$message))
  })
}

# ============================================================================
# ASD-only analysis: Test age effects within ASD group
# ============================================================================
message("\n=== Running ASD-only analysis ===\n")

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_age_data.csv")
  
  if (!file.exists(data_path)) {
    message(paste("Skipping cluster", cluster, "- no data file"))
    next
  }
  
  data <- read_csv(data_path, show_col_types = FALSE)
  
  # Filter for ASD group only
  asd_data <- data %>% filter(Group == "asd")
  
  message(paste("\n=== Processing Cluster", cluster, "(ASD only) ==="))
  
  # Prepare data for modeling
  all_columns <- c(roi, categorical_vars, numerical_vars, "Site")
  sub_data <- asd_data[, all_columns]
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
  
  # Ensure numeric columns are numeric
  sub_data[[roi]] <- as.numeric(sub_data[[roi]])
  for (var in numerical_vars) {
    sub_data[[var]] <- as.numeric(sub_data[[var]])
  }
  
  # Build equation without Group (ASD only)
  equation_lmer <- paste(roi, "~ Age + Sex + (1|Site)")
  
  message(paste("Cluster", cluster, "(ASD only) model:", equation_lmer))
  message(paste("Sample size:", nrow(sub_data)))
  
  # Run model
  tryCatch({
    model <- lmer(as.formula(equation_lmer), data = sub_data)
    
    print(summary(model))
    
    # Write results
    out_file <- paste0(output_dir, "cluster-", cluster, "_age_ASDonly_table.csv")
    model_table <- as.data.frame(coef(summary(model)))
    write.csv(model_table, file = out_file, row.names = TRUE)
    message(paste("Saved:", out_file))
    
    # Create plot
    plot_file <- paste0(output_dir, "cluster-", cluster, "_age_ASDonly_plot.png")
    
    plot_data <- data.frame(
      RSFC = sub_data[[roi]],
      Age = sub_data[["Age"]]
    )
    
    n_asd <- nrow(plot_data)
    
    # Create scatter plot with regression line
    p <- ggplot(plot_data, aes(x = Age, y = RSFC)) +
      geom_point(alpha = 0.6, size = 2, color = "#E74C3C") +
      geom_smooth(method = "lm", se = TRUE, color = "#E74C3C", fill = "#E74C3C") +
      labs(
        title = paste("Cluster", cluster, "- Age Effect (ASD only)"),
        subtitle = sprintf("N = %d", n_asd),
        x = "Age (years)",
        y = "Habenula Connectivity"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30")
      )
    
    ggsave(plot_file, p, width = 7, height = 5, dpi = 300)
    message(paste("Saved plot:", plot_file))
    
  }, error = function(e) {
    message(paste("Error in cluster", cluster, "(ASD only):", e$message))
  })
}

# ============================================================================
# TD-only analysis: Test age effects within TD group
# ============================================================================
message("\n=== Running TD-only analysis ===\n")

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_age_data.csv")
  
  if (!file.exists(data_path)) {
    message(paste("Skipping cluster", cluster, "- no data file"))
    next
  }
  
  data <- read_csv(data_path, show_col_types = FALSE)
  
  # Filter for TD group only
  td_data <- data %>% filter(Group == "td")
  
  message(paste("\n=== Processing Cluster", cluster, "(TD only) ==="))
  
  # Prepare data for modeling
  all_columns <- c(roi, categorical_vars, numerical_vars, "Site")
  sub_data <- td_data[, all_columns]
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
  
  # Ensure numeric columns are numeric
  sub_data[[roi]] <- as.numeric(sub_data[[roi]])
  for (var in numerical_vars) {
    sub_data[[var]] <- as.numeric(sub_data[[var]])
  }
  
  # Build equation without Group (TD only)
  equation_lmer <- paste(roi, "~ Age + Sex + (1|Site)")
  
  message(paste("Cluster", cluster, "(TD only) model:", equation_lmer))
  message(paste("Sample size:", nrow(sub_data)))
  
  # Run model
  tryCatch({
    model <- lmer(as.formula(equation_lmer), data = sub_data)
    
    print(summary(model))
    
    # Write results
    out_file <- paste0(output_dir, "cluster-", cluster, "_age_TDonly_table.csv")
    model_table <- as.data.frame(coef(summary(model)))
    write.csv(model_table, file = out_file, row.names = TRUE)
    message(paste("Saved:", out_file))
    
    # Create plot
    plot_file <- paste0(output_dir, "cluster-", cluster, "_age_TDonly_plot.png")
    
    plot_data <- data.frame(
      RSFC = sub_data[[roi]],
      Age = sub_data[["Age"]]
    )
    
    n_td <- nrow(plot_data)
    
    # Create scatter plot with regression line
    p <- ggplot(plot_data, aes(x = Age, y = RSFC)) +
      geom_point(alpha = 0.6, size = 2, color = "#3498DB") +
      geom_smooth(method = "lm", se = TRUE, color = "#3498DB", fill = "#3498DB") +
      labs(
        title = paste("Cluster", cluster, "- Age Effect (TD only)"),
        subtitle = sprintf("N = %d", n_td),
        x = "Age (years)",
        y = "Habenula Connectivity"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30")
      )
    
    ggsave(plot_file, p, width = 7, height = 5, dpi = 300)
    message(paste("Saved plot:", plot_file))
    
  }, error = function(e) {
    message(paste("Error in cluster", cluster, "(TD only):", e$message))
  })
}

message("\nAnalysis complete!")
