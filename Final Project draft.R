library(rstan)
library(forecast)
library(ggplot2)
library(dplyr)
library(bayesplot)
library(loo)

# Configure Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and preprocess data
merged_data <- read.csv("merged_data.csv") %>%
  arrange(country, year) %>%
  group_by(country) %>%
  mutate(
    lagged_expenditure = lag(government_expenditure),  # Create lagged predictor
    gdp_growth_scaled = scale(gdp_growth)              # Standardize GDP growth
  ) %>%
  ungroup()
valid_countries <- unique(merged_data$country)

#filter data for one country
c <- valid_countries[1]  # Use first country for illustration

country_data <- merged_data %>%
  filter(country == c) %>%
  arrange(year) %>%
  select(year, gdp_growth, lagged_expenditure) %>%
  na.omit()

print(head(country_data))

#arrange train and test dataset
split_index <- floor(0.7 * nrow(country_data))
train <- country_data[1:split_index, ]
test <- country_data[(split_index + 1):nrow(country_data), ]

cat("Training years:", train$year[1], "-", train$year[nrow(train)], "\n")
cat("Test years:", test$year[1], "-", test$year[nrow(test)], "\n")

#Fit ARIMA Model
arima_model <- auto.arima(ts(train$gdp_growth, frequency = 1))
arima_fc <- forecast(arima_model, h = nrow(test))

print(arima_fc)
plot(arima_fc, main = paste("ARIMA Forecast for", c))
lines(test$gdp_growth, col = "red", type = "o")


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

#Fit Bayesian Model in Stan
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

print(stan_fit, pars = c("alpha", "beta", "sigma_obs", "sigma_trend"))

#Posterior Forecasts and intervals
posterior <- extract(stan_fit)

bayes_fc <- apply(posterior$y_forecast, 2, median)
bayes_interval <- apply(posterior$y_forecast, 2, quantile, probs = c(0.025, 0.975))

# Plot Bayesian forecast
plot(1:length(bayes_fc), bayes_fc, type = "l", ylim = range(bayes_interval, test$gdp_growth),
     main = paste("Bayesian Forecast for", c), xlab = "Time", ylab = "GDP Growth")
lines(1:length(bayes_fc), test$gdp_growth, col = "red")
lines(1:length(bayes_fc), bayes_interval[1,], col = "blue", lty = 2)
lines(1:length(bayes_fc), bayes_interval[2,], col = "blue", lty = 2)
legend("topleft", legend = c("Median Forecast", "95% Interval", "Actual"), 
       col = c("black", "blue", "red"), lty = c(1, 2, 1))


#Model Evaluation
# Accuracy Metrics
acc_arima <- accuracy(arima_fc, test$gdp_growth)
acc_bayes <- c(
  RMSE = sqrt(mean((bayes_fc - test$gdp_growth)^2)),
  MAE = mean(abs(bayes_fc - test$gdp_growth)),
  MAPE = mean(abs((bayes_fc - test$gdp_growth)/test$gdp_growth))
)

print(acc_arima["Test set", ])
print(acc_bayes)

#Coverage Metrics
coverage_arima <- mean(test$gdp_growth >= arima_fc$lower[,2] &
                         test$gdp_growth <= arima_fc$upper[,2])

coverage_bayes <- mean(test$gdp_growth >= bayes_interval[1,] &
                         test$gdp_growth <= bayes_interval[2,])

cat("ARIMA Coverage:", round(coverage_arima, 3), "\n")
cat("Bayesian Coverage:", round(coverage_bayes, 3), "\n")

#leave-one-out Cross Validation
log_lik <- extract_log_lik(stan_fit)
loo_score <- loo(log_lik)

print(loo_score)

#Store result in list
single_result <- list(
  country = c,
  arima = arima_fc,
  bayes = list(
    fit = stan_fit,
    forecast = bayes_fc,
    interval = bayes_interval
  ),
  accuracy = list(arima = acc_arima["Test set", ], bayes = acc_bayes),
  coverage = list(arima = coverage_arima, bayes = coverage_bayes),
  loo = loo_score
)

print(single_result)


