library(rstan)
library(forecast)
library(ggplot2)
library(dplyr)
library(bayesplot)
library(loo)

# Configure Stan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#Sorting countries from ARIMA model
#predict by using the ARIMA model 
merged_data <- read.csv("merged_data.csv")

# Identify countries only with >= 36 years of data
valid_countries <- merged_data %>%
  group_by(country) %>%
  filter(n() >= 36) %>%
  distinct(country) %>%
  pull(country)

print(valid_countries)

#Set up the results list for each country in ARIMA model 
results <- list()

for (c in valid_countries) {
  ts_data <- merged_data %>%
    filter(country == c) %>%
    arrange(year)
  
  gdp_ts <- ts(ts_data$gdp_growth, start = min(ts_data$year), frequency = 1)
  
  # Fit ARIMA model by auto regression command
  tryCatch({
    model <- auto.arima(gdp_ts)
    results[[c]] <- model
  }, error = function(e) {
    message("Failed to fit model for ", c, ": ", e$message)
  })
}

#   access selected_models 
selected_models <- results[valid_countries]
lapply(selected_models, summary)

get_order <- function(m) arimaorder(m)
model_orders <- sapply(selected_models, get_order)
print(t(model_orders))  # Transpose for readability

#  Forecast and Accuracy Evaluation
forecast_results <- list()
accuracy_results <- data.frame()

for (c in valid_countries) {
  ts_data <- merged_data %>%
    filter(country == c) %>%
    arrange(year)
  
  #We use the first 30 data as the training data, and the rest for the test
  
  if (nrow(ts_data) < 36) {
    message("Skipping ", c, ": not enough data")
    next
  }
  
  gdp_ts <- ts(ts_data$gdp_growth, start = min(ts_data$year), frequency = 1)
  train <- window(gdp_ts, end = time(gdp_ts)[30])
  test <- window(gdp_ts, start = time(gdp_ts)[31])

  # Showing the accuracy results of ARIMA modle
  tryCatch({
    model <- auto.arima(train)
    fc <- forecast(model, h = length(test))
    forecast_results[[c]] <- fc
    
    acc <- accuracy(fc, test)
    acc_df <- data.frame(
      Country = c,
      RMSE = acc["Test set", "RMSE"],
      MAE = acc["Test set", "MAE"],
      MAPE = acc["Test set", "MAPE"],
      Model = paste0("ARIMA(", paste(arimaorder(model), collapse = ","), ")")
    )
    accuracy_results <- rbind(accuracy_results, acc_df)
  }, error = function(e) {
    message("Forecast failed for ", c, ": ", e$message)
  })
}

print(accuracy_results)

# Check for the coverage of 95% prediction interval 
coverage_results <- data.frame()

for (country in names(forecast_results)) {
  fc <- forecast_results[[country]]
  
  #extract actual test values
  test_actual <- as.numeric(fc$x)[(length(fc$x) - length(fc$mean) + 1):length(fc$x)]
  
  lower <- fc$lower[, "95%"]
  upper <- fc$upper[, "95%"]
  
  #use the actual data check for the coverage
  covered <- test_actual >= lower & test_actual <= upper
  coverage_rate <- mean(covered, na.rm = TRUE)
  
  coverage_results <- rbind(
    coverage_results,
    data.frame(
      Country = country,
      Coverage_95 = coverage_rate,
      Width_95 = mean(upper - lower)
    )
  )
}

print(coverage_results)

#check for PIT Histogram 
pit_values <- numeric()

for (country in names(forecast_results)) {
  fc <- forecast_results[[country]]
  
  var_est <- fc$sigma2
  if (is.null(var_est) || is.na(var_est) || !is.numeric(var_est)) {
    var_est <- fc$model$sigma2
  }
  if (is.null(var_est) || is.na(var_est) || !is.numeric(var_est)) {
    message("Skipping ", country, ": invalid variance (both sigma2 and model$sigma2)")
    next
  }
  
  test_actual <- as.numeric(fc$x)[(length(fc$x) - length(fc$mean) + 1):length(fc$x)]
  pit <- pnorm(test_actual, mean = fc$mean, sd = sqrt(var_est))
  pit_values <- c(pit_values, pit)
}

hist(pit_values, breaks = 10, main = "PIT Histogram", xlab = "Probability Integral Transform")
abline(h = length(pit_values)/10, col = "red", lty = 2)
abline(h = length(pit_values) * 0.095, col = "blue", lty = 3)


#Using 4 countries: United States, New Zealand, Venezuela and Equatorial Guinea to analyze

# Load and preprocess data
merged_data <- merged_data %>%
  arrange(country, year) %>%
  group_by(country) %>%
  mutate(
    lagged_expenditure = lag(government_expenditure),  # Create lagged predictor
    gdp_growth_scaled = scale(gdp_growth)              # Standardize GDP growth
  ) %>%
  ungroup()
#according to the results in ts, we choose 2 with high RMSE and 2 with low RMSE 
valid_countries <- c("United States", "New Zealand", "Venezuela", "Equatorial Guinea")

# Store results
results <- list()
metrics <- data.frame()

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

# Loop through first 20 countries
for (c in valid_countries) {
  tryCatch({
    cat("\n\nProcessing country:", c, "\n")
    
    country_data <- merged_data %>%
      filter(country == c) %>%
      arrange(year) %>%
      select(year, gdp_growth, lagged_expenditure) %>%
      na.omit()
    
    if (nrow(country_data) < 10) {
      cat("Insufficient data for", c, "\n")
      next
    }

    # 70% for training, 30% for testing
    split_index <- floor(0.7 * nrow(country_data))
    train <- country_data[1:split_index, ]
    test <- country_data[(split_index + 1):nrow(country_data), ]
    
    cat("Training years:", train$year[1], "-", train$year[nrow(train)], "\n")
    cat("Test years:", test$year[1], "-", test$year[nrow(test)], "\n")
    
    arima_model <- auto.arima(ts(train$gdp_growth, frequency = 1))
    arima_fc <- forecast(arima_model, h = nrow(test))
    cat("ARIMA forecast complete for", c, "\n")
    
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
      control = list(adapt_delta = 0.95),
      refresh = 0
    )
    cat("Stan model fit complete for", c, "\n")
    
    posterior <- extract(stan_fit)
    bayes_fc <- apply(posterior$y_forecast, 2, median)
    bayes_interval <- apply(posterior$y_forecast, 2, quantile, probs = c(0.025, 0.975))
    
    acc_arima <- accuracy(arima_fc, test$gdp_growth)
    acc_bayes <- c(
      RMSE = sqrt(mean((bayes_fc - test$gdp_growth)^2)),
      MAE = mean(abs(bayes_fc - test$gdp_growth)),
      MAPE = mean(abs((bayes_fc - test$gdp_growth)/test$gdp_growth))
    )
    
    coverage_arima <- mean(test$gdp_growth >= arima_fc$lower[,2] &
                             test$gdp_growth <= arima_fc$upper[,2])
    coverage_bayes <- mean(test$gdp_growth >= bayes_interval[1,] &
                             test$gdp_growth <= bayes_interval[2,])
    
    log_lik <- extract_log_lik(stan_fit)
    loo_score <- loo(log_lik)
    
    metrics <- bind_rows(metrics,
                         data.frame(
                           Country = c,
                           Model = "ARIMA",
                           RMSE = acc_arima["Test set", "RMSE"],
                           MAE = acc_arima["Test set", "MAE"],
                           MAPE = acc_arima["Test set", "MAPE"],
                           Coverage = coverage_arima,
                           CI_Width = mean(arima_fc$upper[,2] - arima_fc$lower[,2]),
                           LOOIC = NA
                         ),
                         data.frame(
                           Country = c,
                           Model = "Bayesian",
                           RMSE = acc_bayes["RMSE"],
                           MAE = acc_bayes["MAE"],
                           MAPE = acc_bayes["MAPE"],
                           Coverage = coverage_bayes,
                           CI_Width = mean(bayes_interval[2,] - bayes_interval[1,]),
                           LOOIC = loo_score$estimates["looic", "Estimate"]
                         )
    )
    
    results[[c]] <- list(
      arima = arima_fc,
      bayes = list(
        fit = stan_fit,
        forecast = bayes_fc,
        interval = bayes_interval
      ),
      loo = loo_score
    )
    
    cat("Modeling complete for", c, "\n")
    
  }, error = function(e) {
    message("Error processing ", c, ": ", e$message)
  })
}




# Calibration Plot: Prediction Interval Coverage Comparison
library(ggplot2)

ggplot(metrics, aes(x = Model, y = Coverage, fill = Model)) +
  geom_boxplot() +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  labs(
    title = "Prediction Interval Coverage Comparison",
    y = "Coverage",
    x = "Model") +
  theme_minimal()

#Trace Plot
trace<-mcmc_trace(
  as.array(results[[1]]$bayes$fit),
  pars = c("alpha", "beta", "sigma_obs")
)
trace
#Posterior Distributions for Key Parameters
posterior_dist<-mcmc_areas(
  as.array(results[[1]]$bayes$fit),
  pars = c("alpha", "beta"),
  prob = 0.95
)

posterior_dist
