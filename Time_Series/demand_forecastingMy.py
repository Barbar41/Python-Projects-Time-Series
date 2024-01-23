####################### ###
#DemandForecasting
####################### ###

# Store Item Demand Forecasting Challenge
# https://www.kaggle.com/c/demand-forecasting-kernels-only
# !pip install lightgbm
# conda install lightgbm
# Developing a demand forecast model for store product breakdown

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings


#############################
# Loading the data
#############################

train = pd.read_csv('Time_Series/datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('Time_Series/datasets/demand_forecasting/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('Time_Series/datasets/demand_forecasting/sample_submission.csv')

df = pd.concat([train, test], sort=False)

def check_df(dataframe, head=5):
     print("######################### Shape #######################")
     print(dataframe.shape)
     print("######################### Types #######################")
     print(dataframe.dtypes)
     print("######################### Head #######################")
     print(dataframe.head(head))
     print("######################### Tail ######################")
     print(dataframe.tail(head))
     print("#########################NA ######################")
     print(dataframe.isnull().sum())
     print("######################### Quantiles #######################")
     print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)



####################### ###
#EDA
####################### ###

df["date"].min(), df["date"].max()

check_df(df)

# How is the sales distribution?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# How many Stores are there
df[["store"]].nunique()

# How many items are there
df[["item"]].nunique()

# We groupby according to the store and check the unique value of the items (Is there an equal number of unique items in each store?
df.groupby(["store"])["item"].nunique()

# You can buy groupby and get the sum of the sales according to the store and the item. How many of the products in the store have been sold?
# Are there equal numbers of sales in each store?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# Let's evaluate the sales statistics in store product breakdowns. Store performances in terms of various products.
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.head()




####################### ###
# FEATURE ENGINEERING
####################### ###

df.head()


def create_date_features(df):
     df['month'] = df.date.dt.month
     df['day_of_month'] = df.date.dt.day
     df['day_of_year'] = df.date.dt.dayofyear
     # Check if 'week_of_year' column exists before converting its data type
     if 'week_of_year' in df.columns:
         df['week_of_year'] = df['week_of_year'].astype(int)
     df['day_of_week'] = df.date.dt.dayofweek
     df['year'] = df.date.dt.year
     df["is_wknd"] = df.date.dt.weekday // 4
     df['is_month_start'] = df.date.dt.is_month_start.astype(int)
     df['is_month_end'] = df.date.dt.is_month_end.astype(int)

     return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

print(df.columns)

#############################
# Random Noise
#############################

# We add random noise to prevent overlearning
def random_noise(dataframe):
     return np.random.normal(scale=1.6, size=(len(dataframe),))


#############################
# Lag/Shifted Features
#############################
# Producing features related to past sales numbers with delay features.
# We create the sales values in previous periods. The data must be sorted according to store product and date.
# So that we can calculate the delay values correctly.
# Therefore, we sort the store by product and date with the sort_values method.
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# We bring the top 10 real values of sales.
pd.DataFrame({"sales": df["sales"].values[0:10],
               "lag1": df["sales"].shift(1).values[0:10],
               "lag2": df["sales"].shift(2).values[0:10],
               "lag3": df["sales"].shift(3).values[0:10],
               "lag4": df["sales"].shift(4).values[0:10]})

# Let's perform this evaluation for a sale in the store product breakdown
df.groupby(["store", "item"])['sales'].head()

# We automatically apply the shift method using the Lambda method with the Transform method.
df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))

# Let's define a function. We will enter different delay values for the features derived for the past.
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
# It will produce new features by entering various intervals and keep them for tracking.
# And noise will be added to them

check_df(df)
df.shape
#############################
# Rolling Mean Features
#############################
# Rolling method to produce moving average features
pd.DataFrame({"sales": df["sales"].values[0:10],
               "roll2": df["sales"].rolling(window=2).mean().values[0:10],
               "roll3": df["sales"].rolling(window=3).mean().values[0:10],
               "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

# We need to receive it after receiving a delay.
pd.DataFrame({"sales": df["sales"].values[0:10],
               "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
               "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
               "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

# Let's get the function.
def roll_mean_features(dataframe, windows):
     for window in windows:
         dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                           transform(
             lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
             dataframe)
     return dataframe

# We are trying to enter information for one year and one and a half years later.
df = roll_mean_features(df, [365, 546])

#############################
# Exponentially Weighted Mean Features
#############################
# We produce average features with exponential weights.
pd.DataFrame({"sales": df["sales"].values[0:10],
               "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
               "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
               "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
               "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
               "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

Let's get a # function.
def ewm_features(dataframe, alphas, lags):
     for alpha in alphas:
         for lag in lags:
             dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                 dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
     return dataframe

# Let's enter the alpha and delay sets.
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]


df = ewm_features(df, alphas, lags)
check_df(df)

# Go to the 90-day delay. When it comes to delays before the 90-day delay, focus on this 90-day delay first.
# When we give more weight to past closer values, more successful predictions are made within the scope of machine learning
# We tried to reflect it in the model by creating variables.

#############################
# One-Hot Encoding
#############################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


check_df(df)


#############################
# Converting sales to log(1+sales)
#############################

# We take log of Sales variable.
df['sales'] = np.log1p(df["sales"].values)

check_df(df)

####################### ###
#Model
####################### ###

#############################
# Custom Cost Function
#############################

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE) (the lower the better)

# Function definition
def smape(preds, target):
     n = len(preds)
     masked_arr = ~((preds == 0) & (target == 0))
     preds, target = preds[masked_arr], target[masked_arr]
     num = np.abs(preds - target)
     denom = np.abs(preds) + np.abs(target)
     smape_val = (200 * np.sum(num / denom)) / n
     return smape_val



# We will use the above function with the Lightgbm algorithm
def lgbm_smape(preds, train_data):
     labels = train_data.get_label()
     smape_val = smape(np.expm1(preds), np.expm1(labels))
     return 'SMAPE', smape_val, False


# Comparison of actual values and predicted values


#############################
# Time-Based Validation Sets
#############################

train
test

# Train set until early 2017 (end 2016).
train = df.loc[(df["date"] < "2017-01-01"), :]

# We will be building a model with Train and verifying it with validation.

# Validation set for the first 3 months of 2017.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Selected all features and independent variables
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales'] # dependent variable (sales) set
X_train = train[cols] # set of arguments

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

#############################
# Time Series Model with LightGBM
#############################

# !pip install lightgbm
# conda install lightgbm


# LightGBM parameters
lgb_params = {'num_leaves': 10,
               'learning_rate': 0.02,
               'feature_fraction': 0.8,
               'max_depth': 5,
               'verbose': 0,
               'num_boost_round': 1000,
               'early_stopping_rounds': 200,
               'nthread':-1}


# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: maximum number of leaves in a tree
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf's random subspace feature. random number of variables to consider in each iteration.
# max_depth: maximum depth
# num_boost_round: n_estimators, number of boosting iterations. It is necessary to make at least around 10000-15000.

# early_stopping_rounds:if the metric in the validation set does not progress in a particular early_stopping_rounds
# In other words, if the error does not occur, stop modeling. It both shortens the train time and prevents overfit.
# nthread: num_thread, nthread, nthreads, n_jobs

# Let's create the data set
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

# Let's create a validation set
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


model = lgb.train(lgb_params, lgbtrain,
                   valid_sets=[lgbtrain, lgbval],
                   num_boost_round=lgb_params['num_boost_round'],
                   early_stopping_rounds=lgb_params['early_stopping_rounds'],
                   feval=lgbm_smape,
                   verbose_eval=100)




# Estimation
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# The log was taken, we took it back and observed the errors
smape(np.expm1(y_pred_val), np.expm1(Y_val))



#############################
# Variable Severity Levels
#############################

# We will extract the feature importances and after converting them into a readable df
# If we want, we will only examine the df, or if we want, we will look at the variable importance levels through a graphical image.
def plot_lgb_importances(model, plot=False, num=10):
     gain = model.feature_importance('gain')
     feat_imp = pd.DataFrame({'feature': model.feature_name(),
                              'split': model.feature_importance('split'),
                              'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
     if plot:
         plt.figure(figsize=(10, 10))
         sns.set(font_scale=1)
         sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
         plt.title('feature')
         plt.tight_layout()
         plt.show()
     else:
         print(feat_imp.head(num))
     return feat_imp

# Number of variables we want to display
plot_lgb_importances(model, num=200)

#Let's visualize
plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


#############################
# Final Model
#############################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
               'learning_rate': 0.02,
               'feature_fraction': 0.8,
               'max_depth': 5,
               'verbose': 0,
               'nthread': -1,
               "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# The logarithmed values are the test_preds values above.

#############################
# Submission File
#############################

test.head()

# ids and sales for the test set
submission_df = test.loc[:, ["id", "sales"]]

# let's place the values we estimated ourselves (we take the log inverse) those in the 90-day period
submission_df['sales'] = np.expm1(test_preds)

# We define the id.
submission_df['id'] = submission_df.id.astype(int)

#

submission_df.to_csv("submission_demand.csv", index=False



