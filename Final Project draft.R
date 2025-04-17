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

#Conduct Stan Model of Bayesian Structural Time Series

stan_model_code <- "
data {
  int<lower=0> T;        // Number of time points
  vector[T] y;           // Response variable (GDP growth)
  vector[T] x;           // Predictor (lagged expenditure)
  int<lower=0> T_forecast; // Forecast horizon
}

parameters {
  real alpha;             // Intercept
  real beta;              // Coefficient for expenditure
  real<lower=0> sigma_obs; // Observation error
  vector[T] trend;        // Local level trend
  real<lower=0> sigma_trend; // Trend innovation
}

model {
  // Priors
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma_obs ~ student_t(3, 0, 1);
  sigma_trend ~ student_t(3, 0, 1);
  
  // State equation (trend evolution)
  trend[1] ~ normal(y[1], sigma_trend);
  for(t in 2:T) {
    trend[t] ~ normal(trend[t-1], sigma_trend);
  }
  
  // Observation equation
  y ~ normal(alpha + beta*x + trend, sigma_obs);
}

generated quantities {
  vector[T_forecast] y_forecast;
  vector[T] log_lik;
  
  // Forecast future values
  y_forecast[1] = normal_rng(alpha + beta*x[T] + trend[T], sigma_obs);
  for(t in 2:T_forecast) {
    y_forecast[t] = normal_rng(alpha + beta*x[T] + trend[T], sigma_obs);
  }
  
  // Compute log-likelihood for LOO
  for(t in 1:T) {
    log_lik[t] = normal_lpdf(y[t] | alpha + beta*x[t] + trend[t], sigma_obs);
  }
}
"