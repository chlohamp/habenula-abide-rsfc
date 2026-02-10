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
clusters <- c("1", "2")
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

# Calculate and save sample sizes for each phenotype
message("\n=== Phenotype Sample Sizes (Unique Subjects) ===")
sample_sizes <- data.frame()

for (phen_var in c("Phen1", "Phen2", "Phen3", "Phen4")) {
  if (phen_var %in% colnames(pheno_data)) {
    # Count unique subjects with non-NA phenotype values
    n_total <- pheno_data %>%
      filter(!is.na(!!sym(phen_var))) %>%
      distinct(Subject) %>%
      nrow()
    
    # Count by group
    n_by_group <- pheno_data %>%
      filter(!is.na(!!sym(phen_var))) %>%
      distinct(Subject, Group) %>%
      group_by(Group) %>%
      summarise(n = n(), .groups = 'drop')
    
    n_asd <- ifelse("asd" %in% n_by_group$Group, n_by_group$n[n_by_group$Group == "asd"], 0)
    n_td <- ifelse("td" %in% n_by_group$Group, n_by_group$n[n_by_group$Group == "td"], 0)
    
    message(sprintf("%s: %d subjects (ASD: %d, TD: %d)", 
                    phen_var, 
                    n_total,
                    n_asd,
                    n_td))
    
    # Add to sample sizes dataframe
    sample_sizes <- rbind(sample_sizes, data.frame(
      Phenotype = phen_var,
      Total_N = n_total,
      ASD_N = n_asd,
      TD_N = n_td
    ))
  } else {
    message(sprintf("%s: Not found in data", phen_var))
  }
}

# Save sample sizes to CSV
sample_sizes_file <- paste0(output_dir, "sample_sizes.csv")
write_csv(sample_sizes, sample_sizes_file)
message(paste("\nSaved sample sizes to:", sample_sizes_file))
message("==============================================\n")

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

# Store p-values for multiple comparison correction
interaction_results <- data.frame()
asd_only_results <- data.frame()

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

      # Ensure Site is a factor and remove NAs
      sub_data$Site <- as.factor(sub_data$Site)
      sub_data <- sub_data[!is.na(sub_data$Site), ]
      # Skip if not enough Site levels for random effect
      if (length(unique(sub_data$Site)) < 2) {
        message(paste("Skipping", phen_var, "in cluster", cluster, "- not enough Site levels for random effect"))
        next
      }
    
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
    # Match original 3dLMEr model structure: use Site as random effect
    fixed_effects <- paste(c(numerical_vars), collapse = " + ")
    equation_lmer <- paste(roi, "~", fixed_effects, "+", group_var, "*", phen_var, "+ (1|Site)")
    
    message(paste("Cluster", cluster, "-", phen_var, ":", equation_lmer))
    
    # Run model with lmer (Site as random effect)
    tryCatch({
      model <- lmer(as.formula(equation_lmer), data = sub_data)
      print(summary(model))
      model_table <- as.data.frame(coef(summary(model)))
      interaction_term <- paste0(group_var, "td:", phen_var)
      interaction_pval <- NA
      if (interaction_term %in% rownames(model_table)) {
        interaction_pval <- model_table[interaction_term, "Pr(>|t|)"]
        interaction_results <- rbind(interaction_results, data.frame(
          Cluster = cluster,
          Phenotype = phen_var,
          N = nrow(sub_data),
          Estimate = model_table[interaction_term, "Estimate"],
          Std_Error = model_table[interaction_term, "Std. Error"],
          t_value = model_table[interaction_term, "t value"],
          p_value = interaction_pval,
          stringsAsFactors = FALSE
        ))
      }
      out_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_table.csv")
      write.csv(model_table, file = out_file, row.names = TRUE)
      message(paste("Saved:", out_file))
      # Only plot if p < 0.05
      if (!is.na(interaction_pval) && interaction_pval < 0.05) {
        plot_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_plot.png")
        plot_data <- data.frame(
          RSFC = sub_data[[roi]],
          Phenotype = sub_data[[phen_var]],
          Group = sub_data[[group_var]]
        )
        n_asd <- sum(plot_data$Group == "asd")
        n_td <- sum(plot_data$Group == "td")
        n_total <- nrow(plot_data)
        # Calculate slopes for each group
        slope_asd <- NA
        slope_td <- NA
        if (sum(plot_data$Group == "asd") > 1) {
          slope_asd <- coef(lm(RSFC ~ Phenotype, data = plot_data[plot_data$Group == "asd", ]))[2]
        }
        if (sum(plot_data$Group == "td") > 1) {
          slope_td <- coef(lm(RSFC ~ Phenotype, data = plot_data[plot_data$Group == "td", ]))[2]
        }
        slope_label <- sprintf("Slope ASD: %.3f\nSlope TD: %.3f", slope_asd, slope_td)
        p <- ggplot(plot_data, aes(x = Phenotype, y = RSFC, color = Group, fill = Group, linetype = Group, shape = Group)) +
          geom_point(alpha = 0.6, size = 1.8) +
          geom_smooth(method = "lm", se = TRUE) +
          scale_color_manual(values = c("asd" = "#b6d191", "td" = "#87B2EA")) +
          scale_fill_manual(values = c("asd" = "#b6d191", "td" = "#87B2EA")) +
          scale_linetype_manual(values = c("asd" = "solid", "td" = "dashed")) +
          scale_shape_manual(values = c("asd" = 16, "td" = 4)) +
          coord_cartesian(ylim = c(-0.3, 0.3)) +
          labs(
            title = paste("Cluster", cluster, "-", phen_var),
            subtitle = sprintf("N = %d (ASD: %d, TD: %d)\n%s", n_total, n_asd, n_td, slope_label),
            x = phen_var,
            y = "Habenula Connectivity"
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30"),
            legend.key.width = unit(2.5, "cm"),
            panel.grid.minor = element_blank()
          )
        ggsave(plot_file, p, width = 7, height = 5, dpi = 300)
        message(paste("Saved plot:", plot_file))
      } else {
        message("Plot not saved: interaction p >= 0.05 or not found.")
      }
    }, error = function(e) {
      message(paste("Error in cluster", cluster, phen_var, ":", e$message))
    })
  }
}

# ============================================================================
# ASD-only analysis: Test phenotype effects within ASD group
# ============================================================================
message("\n=== Running ASD-only analysis ===\n")

for (cluster in clusters) {
  data_path <- paste0(output_dir, "cluster-", cluster, "_data.csv")
  data <- read_csv(data_path, show_col_types = FALSE)
  
  # Filter for ASD group only
  asd_data <- data %>% filter(Group == "asd")
  
  for (phen_var in phen_vars) {
    # Skip if phenotype column doesn't exist or is all NA
    if (!(phen_var %in% colnames(asd_data)) || all(is.na(asd_data[[phen_var]]))) {
      message(paste("Skipping", phen_var, "in cluster", cluster, "- no data"))
      next
    }
    
    all_columns <- c(roi, categorical_vars, numerical_vars, phen_var, "Site")
    sub_data <- asd_data[, all_columns]
    sub_data <- na.omit(sub_data)

      # Ensure Site is a factor and remove NAs
      sub_data$Site <- as.factor(sub_data$Site)
      sub_data <- sub_data[!is.na(sub_data$Site), ]
      # Skip if not enough Site levels for random effect
      if (length(unique(sub_data$Site)) < 2) {
        message(paste("Skipping", phen_var, "in cluster", cluster, "(ASD only) - not enough Site levels for random effect"))
        next
      }
    
    # Skip if insufficient data
    if (nrow(sub_data) < 10) {
      message(paste("Skipping", phen_var, "in cluster", cluster, "- insufficient data"))
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
    sub_data[[phen_var]] <- as.numeric(sub_data[[phen_var]])
    
    # Build equation without Group (ASD only)
    fixed_effects <- paste(c(numerical_vars, phen_var), collapse = " + ")
    equation_lmer <- paste(roi, "~", fixed_effects, "+ (1|Site)")
    
    message(paste("Cluster", cluster, "-", phen_var, "(ASD only):", equation_lmer))
    
    # Run model
    tryCatch({
      model <- lmer(as.formula(equation_lmer), data = sub_data)
      print(summary(model))
      model_table <- as.data.frame(coef(summary(model)))
      phen_pval <- NA
      if (phen_var %in% rownames(model_table)) {
        phen_pval <- model_table[phen_var, "Pr(>|t|)"]
        asd_only_results <- rbind(asd_only_results, data.frame(
          Cluster = cluster,
          Phenotype = phen_var,
          N = nrow(sub_data),
          Estimate = model_table[phen_var, "Estimate"],
          Std_Error = model_table[phen_var, "Std. Error"],
          t_value = model_table[phen_var, "t value"],
          p_value = phen_pval,
          stringsAsFactors = FALSE
        ))
      }
      out_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_ASDonly_table.csv")
      write.csv(model_table, file = out_file, row.names = TRUE)
      message(paste("Saved:", out_file))
      # Only plot if p < 0.05
      if (!is.na(phen_pval) && phen_pval < 0.05) {
        plot_file <- paste0(output_dir, "cluster-", cluster, "_", phen_var, "_ASDonly_plot.png")
        plot_data <- data.frame(
          RSFC = sub_data[[roi]],
          Phenotype = sub_data[[phen_var]]
        )
        n_asd <- nrow(plot_data)
        # Calculate slope for ASD group
        slope_asd <- NA
        if (n_asd > 1) {
          slope_asd <- coef(lm(RSFC ~ Phenotype, data = plot_data))[2]
        }
        slope_label <- sprintf("Slope ASD: %.3f", slope_asd)
        p <- ggplot(plot_data, aes(x = Phenotype, y = RSFC)) +
          geom_point(alpha = 0.6, size = 1.8, color = "#b6d191", shape = 16) +
          geom_smooth(method = "lm", se = TRUE, color = "#b6d191", fill = "#b6d191") +
          coord_cartesian(ylim = c(-0.3, 0.3)) +
          labs(
            title = paste("Cluster", cluster, "-", phen_var, "(ASD only)"),
            subtitle = sprintf("N = %d\n%s", n_asd, slope_label),
            x = phen_var,
            y = "Habenula Connectivity"
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray30"),
            legend.key.width = unit(2.5, "cm"),
            panel.grid.minor = element_blank()
          )
        ggsave(plot_file, p, width = 6, height = 5, dpi = 300)
        message(paste("Saved plot:", plot_file))
      } else {
        message("Plot not saved: ASD-only p >= 0.05 or not found.")
      }
    }, error = function(e) {
      message(paste("Error in cluster", cluster, phen_var, "(ASD only):", e$message))
    })
  }
}

# ============================================================================
# Multiple comparison correction using FDR
# ============================================================================
message("\n=== Applying FDR correction for multiple comparisons ===")

# Correct interaction p-values
if (nrow(interaction_results) > 0) {
  interaction_results$p_fdr <- p.adjust(interaction_results$p_value, method = "fdr")
  interaction_results$significant_fdr <- interaction_results$p_fdr < 0.05
  
  # Save corrected results
  interaction_file <- paste0(output_dir, "interaction_results_fdr_corrected.csv")
  write_csv(interaction_results, interaction_file)
  message(paste("\nSaved FDR-corrected interaction results to:", interaction_file))
  
  message("\n=== Group*Phenotype Interaction Results (FDR-corrected) ===")
  print(interaction_results %>% arrange(p_fdr))
  
  n_sig <- sum(interaction_results$significant_fdr)
  message(sprintf("\nSignificant interactions after FDR correction: %d/%d", n_sig, nrow(interaction_results)))
}

# Correct ASD-only p-values
if (nrow(asd_only_results) > 0) {
  asd_only_results$p_fdr <- p.adjust(asd_only_results$p_value, method = "fdr")
  asd_only_results$significant_fdr <- asd_only_results$p_fdr < 0.05
  
  # Save corrected results
  asd_only_file <- paste0(output_dir, "asd_only_results_fdr_corrected.csv")
  write_csv(asd_only_results, asd_only_file)
  message(paste("\nSaved FDR-corrected ASD-only results to:", asd_only_file))
  
  message("\n=== ASD-only Phenotype Effects (FDR-corrected) ===")
  print(asd_only_results %>% arrange(p_fdr))
  
  n_sig <- sum(asd_only_results$significant_fdr)
  message(sprintf("\nSignificant phenotype effects after FDR correction: %d/%d", n_sig, nrow(asd_only_results)))
}

message("\nAnalysis complete with FDR correction!")

