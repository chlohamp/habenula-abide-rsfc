library("lmerTest")
library("readr")
library("dplyr")
library("tidyr")
library("ggplot2")

# Set paths
data_dir <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/group-drawn/pheno/"
participants_file <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/participants.tsv"
output_dir <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset/group-drawn/pheno/"

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Load cluster data files
clusters <- c("1", "2", "3", "4")
cluster_data_list <- list()

for (cluster in clusters) {
  csv_file <- paste0(data_dir, "pheno-correlation-cluster", cluster, ".csv")
  if (file.exists(csv_file)) {
    cluster_data_list[[cluster]] <- read_csv(csv_file, show_col_types = FALSE)
  } else {
    warning(paste("File not found:", csv_file))
  }
}

# Combine all cluster data
combined_data <- bind_rows(cluster_data_list)

# Remove rows with score == -9999.0
combined_data <- combined_data %>% filter(score != -9999.0)

# Calculate mean Correlation for each Subject, cluster, and phenotype
mean_corr_data <- combined_data %>%
  group_by(Subject, Group, Age, Cluster, phenotype, score) %>%
  summarise(Correlation = mean(Correlation), .groups = 'drop')

# Pivot wider to have phenotypes as columns
pivot_data <- mean_corr_data %>%
  pivot_wider(
    id_cols = c(Subject, Group, Age, Cluster, Correlation),
    names_from = phenotype,
    values_from = score
  )

# Combine related phenotypes (use coalesce to take non-NA value)
pivot_data <- pivot_data %>%
  mutate(
    SRS_MOTIVATION = coalesce(SRS_MOTIVATION, SRS_MOTIVATION_RAW),
    SRS_COMMUNICATION = coalesce(SRS_COMMUNICATION, SRS_COMMUNICATION_RAW),
    VINELAND_DAILYLIVING = coalesce(VINELAND_DAILYLIVING_STANDARD, VINELAND_DAILYLVNG_STANDARD)
  ) %>%
  select(-SRS_MOTIVATION_RAW, -SRS_COMMUNICATION_RAW, -VINELAND_DAILYLIVING_STANDARD, -VINELAND_DAILYLVNG_STANDARD)

# Rename columns to match R script expectations
colnames(pivot_data)[colnames(pivot_data) == "Correlation"] <- "RSFC"
colnames(pivot_data)[colnames(pivot_data) == "SRS_MOTIVATION"] <- "Phen1"
colnames(pivot_data)[colnames(pivot_data) == "SRS_COMMUNICATION"] <- "Phen2"
colnames(pivot_data)[colnames(pivot_data) == "VINELAND_DAILYLIVING"] <- "Phen3"
colnames(pivot_data)[colnames(pivot_data) == "BRIEF_GEC_T"] <- "Phen4"

# Load participants data
participants <- read_tsv(participants_file, show_col_types = FALSE) %>%
  select(participant_id, SITE_ID, SEX) %>%
  rename(Subject = participant_id, Site = SITE_ID, Sex = SEX)

# Merge with participants data
merged_data <- left_join(pivot_data, participants, by = "Subject")

# Reorder columns
ordered_columns <- c("Subject", "Cluster", "Group", "Age", "Sex", "Site", "RSFC",
                     "Phen1", "Phen2", "Phen3", "Phen4")
pheno_data <- merged_data %>% select(any_of(ordered_columns))

# Save cluster-specific CSV files for statistical analysis
for (cluster in clusters) {
  cluster_df <- pheno_data %>% 
    filter(Cluster == cluster) %>%
    select(-Cluster)
  
  output_file <- paste0(output_dir, "cluster-", cluster, "_data.csv")
  write_csv(cluster_df, output_file)
  message(paste("Saved:", output_file))
}

# Run statistical models
roi <- "RSFC"
group_var <- "Group"
categorical_vars <- c("Sex")
numerical_vars <- c("Age")
phen_vars <- c("Phen1", "Phen2", "Phen3", "Phen4")

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_data.csv")
  data <- read_csv(data_path, show_col_types = FALSE)
  
  for (phen_var in phen_vars) {
    # Skip if phenotype column doesn't exist or is all NA
    if (!(phen_var %in% colnames(data)) || all(is.na(data[[phen_var]]))) {
      message(paste("Skipping", phen_var, "in cluster", cluster, "- no data"))
      next
    }
    
    all_columns <- c(roi, categorical_vars, numerical_vars, phen_var, group_var, "Site")
    sub_data <- data[, all_columns]
    sub_data <- na.omit(sub_data)
    
    # Skip if insufficient data
    if (nrow(sub_data) < 10) {
      message(paste("Skipping", phen_var, "in cluster", cluster, "- insufficient data"))
      next
    }
    
    # Convert categorical variables to factors
    for (var in categorical_vars) {
      sub_data[[var]] <- factor(sub_data[[var]])
    }
    
    # Relevel Group to make asd the reference
    sub_data[[group_var]] <- relevel(factor(sub_data[[group_var]]), ref = "asd")
    
    # Ensure numeric columns are numeric (no scaling)
    sub_data[[roi]] <- as.numeric(sub_data[[roi]])
    for (var in numerical_vars) {
      sub_data[[var]] <- as.numeric(sub_data[[var]])
    }
    sub_data[[phen_var]] <- as.numeric(sub_data[[phen_var]])
    
    # Build equation with Group*Phenotype interaction
    fixed_effects <- paste(c(numerical_vars), collapse = " + ")
    equation_lm <- paste(roi, "~", fixed_effects, "+", group_var, "*", phen_var, "+ Site")
    
    message(paste("Cluster", cluster, "-", phen_var, ":", equation_lm))
    
    # Run model with lm (Site as fixed effect to avoid lmer issues)
    tryCatch({
      model <- lm(as.formula(equation_lm), data = sub_data)
      
      print(summary(model))
      
      # Write results
      out_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_table.csv")
      model_table <- as.data.frame(coef(summary(model)))
      write.csv(model_table, file = out_file, row.names = TRUE)
      message(paste("Saved:", out_file))
      
      # Create plot
      plot_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_plot.png")
      
      # Use raw data for plotting with group colors
      plot_data <- data.frame(
        RSFC = sub_data[[roi]],
        Phenotype = sub_data[[phen_var]],
        Group = sub_data[[group_var]]
      )
      
      # Create scatter plot with regression lines by group
      p <- ggplot(plot_data, aes(x = Phenotype, y = RSFC, color = Group, fill = Group)) +
        geom_point(alpha = 0.6, size = 2) +
        geom_smooth(method = "lm", se = TRUE) +
        scale_color_manual(values = c("asd" = "#E74C3C", "td" = "#3498DB")) +
        scale_fill_manual(values = c("asd" = "#E74C3C", "td" = "#3498DB")) +
        labs(
          title = paste("Cluster", cluster, "-", phen_var),
          x = phen_var,
          y = "Habenula Connectivity"
        ) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5))
      
      ggsave(plot_file, p, width = 7, height = 5, dpi = 300)
      message(paste("Saved plot:", plot_file))
      
    }, error = function(e) {
      message(paste("Error in cluster", cluster, phen_var, ":", e$message))
    })
  }
}

message("\nAnalysis complete!")

