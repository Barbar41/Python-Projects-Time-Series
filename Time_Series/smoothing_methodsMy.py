##################################################
# Smoothing Methods (Holt-Winters)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')


#############################
# Data set
#############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

# we call the s module
data = sm.datasets.co2.load_pandas()

# We define our target variable, our dependent variable and our time series data.
y = data.data

# Monthly average according to frequency information
y = y['co2'].resample('MS').mean()

# Missing Values
y.isnull().sum()

# To fill it, we fill it using the next values.
y = y.fillna(y.bfill())

#Let's visualize
y.plot(figsize=(15, 6))
plt.show()

# There is a trend in this series; there is an increase over time.
# This series is not stationary; the series changes over time.
# There is seasonality in this series. It appears to be decreasing.

# We move on to the modeling method.

#############################
#Holdout
#############################
# Models tend to over-learn. To prevent this, we separate the set.

# Training set until this date
train = y[:'1997-12-01']
len(train) # 478 months

# Test set from the first month of 1998 to the end of 2001.
test = y['1998-01-01':]
len(test) # 48 months

#######################
# Time Series Structural Analysis
#######################

# Stationarity Test (Dickey-Fuller Test)

def is_stationary(y):

     # "HO: Non-stationary"
     # "H1: Stationary"

     p_value = sm.tsa.stattools.adfuller(y)[1]
     if p_value < 0.05:
         print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
     else:
         print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)
# There is no stasis.


# Time Series Components and Stationarity Testing

def ts_decompose(y, model="additive", stationary=False):
     result = seasonal_decompose(y, model=model)
     fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
     fig.set_figheight(10)
     fig.set_figwidth(15)

     axes[0].set_title("Decomposition for " + model + " model")
     axes[0].plot(y, 'k', label='Original ' + model)
     axes[0].legend(loc='upper left')

     axes[1].plot(result.trend, label='Trend')
     axes[1].legend(loc='upper left')

     axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
     axes[2].legend(loc='upper left')

     axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
     axes[3].legend(loc='upper left')
     plt.show(block=True)

     if stationary:
         is_stationary(y)

ts_decompose(y, stationary=True)


#######################
# Single Exponential Smoothing
#######################

# SOUND = Level
# We are building a model
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

# Let's keep the prediction results
y_pred = ses_model.forecast(48)

# Let's evaluate our error (compare the results of the prediction with the actual values)
mean_absolute_error(test, y_pred)

# Let's visualize the train set
train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
# Greens are the values we predicted. We also made a failed prediction

# If we want to take a closer look.
train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

# Function that can display both the image and mean_absolute_error at the same time.

def plot_co2(train, test, y_pred, title):
     mae = mean_absolute_error(test, y_pred)
     train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
     test.plot(legend=True, label="TEST", figsize=(6, 4))
     y_pred.plot(legend=True, label="PREDICTION")
     plt.show()

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

# Let's look at the parameters of this model
ses_model.params

#############################
# Hyperparameter Optimization
#############################

def ses_optimizer(train, alphas, step=48):

     best_alpha, best_mae = None, float("inf")

     for alpha in alphas:
         ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
         y_pred = ses_model.forecast(step)
         mae = mean_absolute_error(test, y_pred)

         if mae < best_mae:
             best_alpha, best_mae = alpha, mae

         print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
     print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
     return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)
# Due to the understanding that a weak model whose values are close to the real values may have more successful results, the past will be closer to the future values.

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

# What is wanted to be done is to observe what the error is against different alpha values.
ses_optimizer(train, alphas)

# To ban the squeal result;
best_alpha, best_mae = ses_optimizer(train, alphas)

#############################
# Final SES Model
#############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")


#######################
# Double Exponential Smoothing (DES)
#######################

# DES: Level (SES) + Trend

# y(t) = Level + Trend + Seasonality + Noise "The series is additive if the seasonality and residual components are independent of the trend."

# y(t) = Level * Trend * Seasonality * Noise "The series is multiplicative if the seasonality and residual components are dependent on the trend."


ts_decompose(y)

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                          smoothing_trend=0.5)

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

#############################
# Hyperparameter Optimization
#############################


def des_optimizer(train, alphas, betas, step=48):
     best_alpha, best_beta, best_mae = None, None, float("inf")
     for alpha in alphas:
         for beta in betas:
             des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
             y_pred = des_model.forecast(step)
             mae = mean_absolute_error(test, y_pred)
             if mae < best_mae:
                 best_alpha, best_beta, best_mae = alpha, beta, mae
             print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
     print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
     return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)




#############################
# Final DES Model
#############################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                                smoothing_slope=best_beta)

y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")


#######################
# Triple Exponential Smoothing (Holt-Winters)
#######################
# If there is no seasonality or trend combination, it is the SES method.
# If there is no seasonality component, this is the DES method.
# If there are all other components plus the seasonality component, this is the Holt-Winter method.

# TES = SES + DES + Seasonality
# TES is the most advanced smoothing method.
# This method makes predictions by dynamically evaluating level, trend and seasonality effects
# Single variables containing trend or seasonality are used in series.


tes_model = ExponentialSmoothing(train,
                                  trend="add",
                                  seasonal="add",
                                  seasonal_periods=12).fit(smoothing_level=0.5,
                                                           smoothing_slope=0.5,
                                                           smoothing_seasonal=0.5)

y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

#############################
# Hyperparameter Optimization
#############################

alphas = betas = gammas = np.arange(0.20, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))


def tes_optimizer(train, abg, step=48):
     best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
     for comb in abg:
         tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
             fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
         y_pred = tes_model.forecast(step)
         mae = mean_absolute_error(test, y_pred)
         if mae < best_mae:
             best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
         print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

     print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
           "best_mae:", round(best_mae, 4))

     return best_alpha, best_beta, best_gamma, best_mae

# We take the values that give the best error
best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)




#############################
# Final TES Model
#############################

# Let's see the resulting value as a final;

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")








