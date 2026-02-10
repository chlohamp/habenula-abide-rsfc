library(readr)
library(dplyr)
library(lme4)
library(ggplot2)

dset_dir <- "/Users/chloehampson/Desktop/habenula-abide-rsfc/dset"
participants_file <- file.path(dset_dir, "participants.tsv")
included_particpants_file <- file.path(dset_dir, "group-drawn/habenula/sub-group_task-rest_desc-1S2StTesthabenula_table.txt")

# Load and clean data
df0 <- read_tsv(participants_file, show_col_types = FALSE)
included_df <- read_tsv(included_particpants_file, show_col_types = FALSE)
included_subjects <- included_df$Subj

df <- df0 %>%
  filter(participant_id %in% included_subjects) %>%
  mutate(
    Site = as.factor(SITE_ID),
    Sex = case_when(
      SEX %in% c(1, "1") ~ "M",
      SEX %in% c(2, "2") ~ "F",
      TRUE ~ as.character(SEX)
    ),
    Sex = as.factor(Sex),
    Group = case_when(
      DX_GROUP %in% c(1, "1") ~ "asd",
      DX_GROUP %in% c(2, "2") ~ "td",
      TRUE ~ NA_character_
    ),
    Group = factor(Group, levels = c("asd", "td")),
    Age = as.numeric(AGE_AT_SCAN)
  ) %>%
  filter(!is.na(Site), !is.na(Sex), !is.na(Group), !is.na(Age))

cat("Sample:", nrow(df), "participants across", nlevels(df$Site), "sites\n")
cat("ASD:", sum(df$Group == "asd"), "| TD:", sum(df$Group == "td"), "\n\n")

# ==============================================
# SITE SIMILARITY/VARIABILITY ANALYSIS
# ==============================================

# Calculate site-level summaries
site_summary <- df %>%
  group_by(Site) %>%
  summarise(
    total_n = n(),
    asd_n = sum(Group == "asd"),
    td_n = sum(Group == "td"),
    asd_prop = mean(Group == "asd"),
    male_n = sum(Sex == "M"),
    female_n = sum(Sex == "F"),
    male_prop = mean(Sex == "M"),
    mean_age = mean(Age),
    sd_age = sd(Age),
    .groups = "drop"
  )

cat("SITE VARIABILITY IN GROUP COMPOSITION:\n")
cat("=====================================\n")

# ASD proportion variability across sites
asd_prop_mean <- mean(site_summary$asd_prop)
asd_prop_sd <- sd(site_summary$asd_prop)
asd_prop_range <- max(site_summary$asd_prop) - min(site_summary$asd_prop)
asd_prop_cv <- asd_prop_sd / asd_prop_mean

cat("ASD proportion across sites:\n")
cat("  Mean:", round(asd_prop_mean, 3), "\n")
cat("  Range:", round(min(site_summary$asd_prop), 3), "to", round(max(site_summary$asd_prop), 3), "\n")
cat("  Standard deviation:", round(asd_prop_sd, 3), "\n")
cat("  Coefficient of variation:", round(asd_prop_cv, 3), "\n")
if(!is.na(asd_prop_range)) {
  if(asd_prop_range > 0.3) {
    cat("  → HIGH variability - sites very different!\n")
  } else if(asd_prop_range > 0.15) {
    cat("  → MODERATE variability\n") 
  } else {
    cat("  → LOW variability - sites similar\n")
  }
} else {
  cat("  → Unable to calculate variability\n")
}

cat("\nSEX proportion across sites:\n")
male_prop_mean <- mean(site_summary$male_prop)
male_prop_sd <- sd(site_summary$male_prop)
male_prop_range <- max(site_summary$male_prop) - min(site_summary$male_prop)
male_prop_cv <- male_prop_sd / male_prop_mean

cat("  Mean male proportion:", round(male_prop_mean, 3), "\n")
cat("  Range:", round(min(site_summary$male_prop), 3), "to", round(max(site_summary$male_prop), 3), "\n")
cat("  Standard deviation:", round(male_prop_sd, 3), "\n")
cat("  Coefficient of variation:", round(male_prop_cv, 3), "\n")

cat("\nAGE across sites:\n")
age_mean_range <- max(site_summary$mean_age) - min(site_summary$mean_age)
age_mean_sd <- sd(site_summary$mean_age)

cat("  Mean age range:", round(min(site_summary$mean_age), 1), "to", round(max(site_summary$mean_age), 1), "years\n")
cat("  SD of site means:", round(age_mean_sd, 2), "years\n")

cat("\nSITE SIZES:\n")
cat("  Range:", min(site_summary$total_n), "to", max(site_summary$total_n), "participants\n")
cat("  Small sites (n<20):", sum(site_summary$total_n < 20), "out of", nrow(site_summary), "\n")

# Statistical test for independence
cat("\nSTATISTICAL TESTS:\n")
cat("==================\n")

# Group x Site
chi_test_group <- chisq.test(table(df$Site, df$Group))
cat("Group × Site independence test:\n")
cat("  χ² =", round(chi_test_group$statistic, 2), "\n")
cat("  p-value =", format(chi_test_group$p.value, scientific=TRUE, digits=3), "\n")
if(!is.na(chi_test_group$p.value)) {
  if(chi_test_group$p.value < 0.001) {
    cat("  → STRONG evidence sites differ in ASD/TD composition\n")
  } else if(chi_test_group$p.value < 0.05) {
    cat("  → Evidence sites differ in ASD/TD composition\n")
  } else {
    cat("  → No evidence sites differ in ASD/TD composition\n")
  }
} else {
  cat("  → Unable to perform chi-square test\n")
}

# Sex x Site
chi_test_sex <- chisq.test(table(df$Site, df$Sex))
cat("\nSex × Site independence test:\n")
cat("  χ² =", round(chi_test_sex$statistic, 2), "\n")
cat("  p-value =", format(chi_test_sex$p.value, scientific=TRUE, digits=3), "\n")
if(!is.na(chi_test_sex$p.value)) {
  if(chi_test_sex$p.value < 0.001) {
    cat("  → STRONG evidence sites differ in sex composition\n")
  } else if(chi_test_sex$p.value < 0.05) {
    cat("  → Evidence sites differ in sex composition\n")
  } else {
    cat("  → No evidence sites differ in sex composition\n")
  }
} else {
  cat("  → Unable to perform chi-square test\n")
}

# Age x Site (ANOVA)
age_anova <- aov(Age ~ Site, data = df)
age_f_stat <- summary(age_anova)[[1]][["F value"]][1]
age_p_value <- summary(age_anova)[[1]][["Pr(>F)"]][1]
cat("\nAge × Site ANOVA:\n")
cat("  F =", round(age_f_stat, 2), "\n")
cat("  p-value =", format(age_p_value, scientific=TRUE, digits=3), "\n")
if(!is.na(age_p_value)) {
  if(age_p_value < 0.001) {
    cat("  → STRONG evidence sites differ in age\n")
  } else if(age_p_value < 0.05) {
    cat("  → Evidence sites differ in age\n")
  } else {
    cat("  → No evidence sites differ in age\n")
  }
} else {
  cat("  → Unable to perform ANOVA\n")
}

cat("\n")

# ==============================================
# VISUALIZATION OF SITE EFFECTS
# ==============================================

# Define colors
group_colors <- c("asd" = "#b6d191", "td" = "#87B2EA")
sex_colors <- c("M" = "#66B2FF", "F" = "#FF9999")

# Plot 1: Age distribution across sites
p1 <- ggplot(df, aes(x = Site, y = Age, fill = Group)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = group_colors) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Age Distribution Across Sites",
    subtitle = paste("Age range:", round(min(site_summary$mean_age), 1), "to", round(max(site_summary$mean_age), 1), "years"),
    x = "Site",
    y = "Age (years)",
    fill = "Group"
  )

# Plot 2: Group distribution across sites
site_group_counts <- df %>%
  group_by(Site, Group) %>%
  count() %>%
  ungroup()

p2 <- ggplot(site_group_counts, aes(x = Site, y = n, fill = Group)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  scale_fill_manual(values = group_colors) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Group Distribution Across Sites",
    subtitle = paste("ASD proportion range:", round(min(site_summary$asd_prop), 2), "to", round(max(site_summary$asd_prop), 2)),
    x = "Site", 
    y = "Number of Participants",
    fill = "Group"
  )

# Plot 3: Sex distribution across sites
site_sex_counts <- df %>%
  group_by(Site, Sex) %>%
  count() %>%
  ungroup()

p3 <- ggplot(site_sex_counts, aes(x = Site, y = n, fill = Sex)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  scale_fill_manual(values = sex_colors) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Sex Distribution Across Sites",
    subtitle = paste("Male proportion range:", round(min(site_summary$male_prop), 2), "to", round(max(site_summary$male_prop), 2)),
    x = "Site", 
    y = "Number of Participants",
    fill = "Sex"
  )

# Display and save plots
print(p1)
ggsave(file.path(dset_dir, "age-sites-barplot.png"), p1, width = 12, height = 8, dpi = 300)

print(p2)
ggsave(file.path(dset_dir, "group-sites-barplot.png"), p2, width = 12, height = 8, dpi = 300)

print(p3)
ggsave(file.path(dset_dir, "sex-sites-barplot.png"), p3, width = 12, height = 8, dpi = 300)
