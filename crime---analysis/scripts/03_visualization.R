# ===============================
# 03_visualization.R
# Crime Visualizations
# ===============================

library(tidyverse)
library(ggplot2)

crime <- read_csv("data/cleaned_crime_data.csv")


# 1. Year-wise crime trend
yearly <- crime %>%
  group_by(YEAR) %>%
  summarise(total = sum(crime_count), .groups = "drop")

ggplot(yearly, aes(YEAR, total)) +
  geom_line(color = "firebrick", linewidth = 1) +
  geom_point() +
  scale_x_continuous(breaks = sort(unique(yearly$YEAR))) +
  labs(
    title = "Year-wise Crime Trend in India",
    x = "Year",
    y = "Total Crimes"
  ) +
  theme_minimal()

# 2. Top 10 states
top_states <- crime %>%
  group_by(`STATE/UT`) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  slice_head(n = 10)

ggplot(top_states,
       aes(x = reorder(`STATE/UT`, total), y = total)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 States by Crime",
    x = "State",
    y = "Total Crimes"
  ) +
  theme_minimal()

# 3. Top 10 crime types
top_crimes <- crime %>%
  group_by(crime_type) %>%
  summarise(total = sum(crime_count), .groups = "drop") %>%
  arrange(desc(total)) %>%
  slice_head(n = 10)

ggplot(top_crimes,
       aes(x = reorder(crime_type, total), y = total)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top 10 Crime Types in India",
    x = "Crime Type",
    y = "Total Crimes"
  ) +
  theme_minimal()
