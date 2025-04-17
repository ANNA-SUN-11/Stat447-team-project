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
#Compare ARIMA and BSTS model

results <- list()
metrics <- data.frame()

for (c in valid_countries[1:5]) {  
  tryCatch({
    country_data <- merged_data %>%
      filter(country == c) %>%
      arrange(year) %>%
      select(year, gdp_growth, lagged_expenditure) %>%
      na.omit()  #Looping through first 5 countries, if it works for 5, it can scale up to all
    
    # Split data
    split_index <- floor(0.7 * nrow(country_data))
    train <- country_data[1:split_index, ]
    test <- country_data[(split_index + 1):nrow(country_data), ]#Set training dataset and testing dataset
    
    # ARIMA Baseline
    arima_model <- auto.arima(ts(train$gdp_growth, frequency = 1))#automatically selects the best ARIMA(p,d,q) model
    arima_fc <- forecast(arima_model, h = nrow(test)) #predicts GDP growth h steps ahead
    
    # Bayesian Model with Stan
    stan_data <- list(
      T = nrow(train),
      y = train$gdp_growth,
      x = train$lagged_expenditure,
      T_forecast = nrow(test)
    )
    
    stan_fit <- stan(
      model_code = stan_model_code,
      data = stan_data,
      iter = 2000,
      chains = 4,
      control = list(adapt_delta = 0.95)
    )