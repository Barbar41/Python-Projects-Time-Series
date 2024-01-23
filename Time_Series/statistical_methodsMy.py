##################################################
# Statistical Methods
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')
import statsmodels.api as smapi

#NOTES#
# AR(p):Autoregression #
# Prediction is made by linear combination of observations in previous time steps.
# Suitable for univariate time series that do not contain trend and seasonality.

# MA(q):Moving Average #
# A linear combination of the errors obtained in previous time steps is estimated.

# ARMA(p,q) = AR(p) + MA(q)
# AutoRegressive Moving Average.Combines AR and MA methods
# Prediction is made by a linear combination of past values and past errors.
# Suitable for univariate time series that do not contain trend and seasonality.
# p and q are time delay numbers. p for AR model, q for MA model



#############################
# Data set
#############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':].resample('MS').mean()
test = test.fillna(test.bfill())


#######################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
#######################
# Prediction is made by a linear combination of differentiated observations and errors from previous time steps.
# Suitable for univariate data that has a trend but no seasonality.
# p: real value lag number (autoregressive degree) If p = 2, yt-1 and yt-2 are in the model.
# d: number of difference operations (degree of difference, l)
# q: number of error delays (moving average degree)

#------------------------------------------------- -----
# arima_model = ARIMA(train, order=(1, 1, 1)).fit()
# arima_model.summary()
#------------------------------------------------- ----
model= smapi.tsa.arima.ARIMA(train, order=(2,1,3))
result = model.fit()
print(result.summary())

y_pred = result.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

def plot_co2(train, test, y_pred, title):
     mae = mean_absolute_error(test, y_pred)
     train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
     test.plot(legend=True, label="TEST", figsize=(6, 4))
     y_pred.plot(legend=True, label="PREDICTION")
     plt.show()

plot_co2(train, test, y_pred, "ARIMA")

#############################
# Hyperparameter Optimization (Determining Model Degrees)
#############################

#############################
# Determining Model Rank Based on AIC & BIC Statistics
#############################

# We create combinations
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

def arima_optimizer_aic(train, orders):
     best_aic, best_params = float("inf"), None
     for order in orders:
         try:
             model = smapi.tsa.arima.ARIMA(train, order=order)
             result = model.fit()
             aic = result.aic
             if aic < best_aic:
                 best_aic, best_params = aic, order
             print('ARIMA%s AIC=%.2f' % (order, aic))
         except:
             continue
     print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
     return best_params

best_params_aic = arima_optimizer_aic(train, pdq)


#############################
# Final Model
#############################
model= smapi.tsa.arima.ARIMA(train, order=best_params_aic)

y_pred = result.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")

####################### ###########
#######################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
#######################

model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))

sarima_model = model.fit(disp=0)

y_pred_test = sarima_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean

y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred)




#############################
# Hyperparameter Optimization (Determining Model Degrees)
#############################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)


############################
# Final Model
############################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred)


#######################
# BONUS: WINDING Optimization by MAE
#######################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, pdq, seasonal_pdq):
     best_mae, best_order, best_seasonal_order = float("inf"), None, None
     for param in pdq:
         for param_seasonal in seasonal_pdq:
             try:
                 model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                 sarima_model = model.fit(disp=0)
                 y_pred_test = sarima_model.get_forecast(steps=48)
                 y_pred = y_pred_test.predicted_mean
                 mae = mean_absolute_error(test, y_pred)
                 if mae < best_mae:
                     best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                 print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
             except:
                 continue
     print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
     return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

def plot_co2(train, test, y_pred, title):
     mae = mean_absolute_error(test, y_pred)
     train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae, 2)}")
     test.plot(legend=True, label="TEST", figsize=(6, 4))
     y_pred.plot(legend=True, label="PREDICTION")
     plt.show()


plot_co2(train, test, y_pred, "SARIMA")


#############################
# Final Model
#############################

model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

feature_predict = sarima_final_model.get_forecast(steps=6)
feature_predict = feature_predict.predicted_mean


