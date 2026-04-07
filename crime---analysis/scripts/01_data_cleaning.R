# ============================================
# 01_data_cleaning.R
# Final Data Preprocessing (NCRB Dataset)
# ============================================

library(tidyverse)

# --------------------------------------------
# Load raw NCRB dataset (wide format)
# --------------------------------------------
crime_raw <- read_csv("data/dataset.csv", show_col_types = FALSE)

# --------------------------------------------
# Identify crime columns (exclude identifiers)
# --------------------------------------------
crime_cols <- setdiff(
  colnames(crime_raw),
  c("STATE/UT", "DISTRICT", "YEAR")
)

# --------------------------------------------
# Convert WIDE → LONG (tidy format)
# --------------------------------------------
crime_clean <- crime_raw %>%
  pivot_longer(
    cols = all_of(crime_cols),
    names_to = "crime_type",
    values_to = "crime_count"
  ) %>%
  mutate(
    crime_count = as.numeric(crime_count)
  ) %>%
  filter(!is.na(crime_count))

# --------------------------------------------
# REMOVE AGGREGATE IPC CATEGORIES (CRITICAL)
# --------------------------------------------
crime_clean <- crime_clean %>%
  filter(!crime_type %in% c(
    "TOTAL IPC CRIMES",
    "OTHER IPC CRIMES"
  ))

# --------------------------------------------
# Final sanity checks (optional)
# --------------------------------------------
stopifnot(
  all(c("STATE/UT", "DISTRICT", "YEAR", "crime_type", "crime_count") %in% colnames(crime_clean)),
  is.numeric(crime_clean$crime_count),
  is.numeric(crime_clean$YEAR)
)

# --------------------------------------------
# Save cleaned dataset (FINAL MASTER FILE)
# --------------------------------------------
write_csv(crime_clean, "data/cleaned_crime_data.csv")
