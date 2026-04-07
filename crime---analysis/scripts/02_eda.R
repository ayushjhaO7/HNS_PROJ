# ===============================
# 02_eda.R
# Exploratory Data Analysis
# ===============================

library(tidyverse)

crime <- read_csv("data/cleaned_crime_data.csv")

crime <- crime %>%
  filter(!crime_type %in% c(
    "TOTAL IPC CRIMES",
    "OTHER IPC CRIMES"
  ))


# Structure
str(crime)

# Dimensions
dim(crime)

# Year range
range(crime$YEAR)
length(unique(crime$YEAR))

# Missing values
colSums(is.na(crime))

# Distribution
summary(crime$crime_count)

# Top states by crime
crime %>%
  group_by(`STATE/UT`) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  head(10)

# Top crime types
crime %>%
  group_by(crime_type) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  head(10)
