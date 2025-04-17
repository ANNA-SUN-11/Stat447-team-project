library(rstan)
library(forecast)
library(dplyr)
library(bayesplot)

# Load and preprocess data
merged_data <- read.csv("merged_data.csv") %>%
  arrange(country, year) %>%
  group_by(country) %>%
  mutate(
    lagged_expenditure = lag(government_expenditure),  # Create lagged predictor
    gdp_growth_scaled = scale(gdp_growth)              # Standardize GDP growth
  ) %>%
  ungroup()

# Filter countries with sufficient data
valid_countries <- merged_data %>%
  group_by(country) %>%
  summarise(n = sum(!is.na(gdp_growth) & !is.na(lagged_expenditure))) %>%
  filter(n >= 36) %>%
  pull(country)