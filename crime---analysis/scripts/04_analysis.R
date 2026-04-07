# ===============================
# 04_analysis.R
# Crime Data Analysis
# ===============================

library(tidyverse)

crime <- read_csv("data/cleaned_crime_data.csv")

crime <- crime %>%
  filter(!crime_type %in% c(
    "TOTAL IPC CRIMES",
    "OTHER IPC CRIMES"
  ))

# Total crimes
sum(crime$crime_count)

# Most crime-prone state
crime %>%
  group_by(`STATE/UT`) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  slice(1)

# Peak crime year
crime %>%
  group_by(YEAR) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  slice(1)

# Crime growth over years
crime %>%
  group_by(YEAR) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(YEAR) %>%
  mutate(
    growth = total - lag(total),
    growth_rate = round((growth / lag(total)) * 100, 2)
  )

# High-risk states
crime %>%
  group_by(`STATE/UT`) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  mutate(
    risk_level = case_when(
      total >= quantile(total, 0.75) ~ "High Risk",
      total >= quantile(total, 0.40) ~ "Medium Risk",
      TRUE ~ "Low Risk"
    )
  ) %>%
  arrange(desc(total))
