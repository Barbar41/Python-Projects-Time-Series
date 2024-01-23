
#################
# Expected Transaction Volume Estimate from Iyzico Member Businesses
##############

# Business Problem:
# Iyzico is a financial technologies company that makes the online shopping experience easier for both buyers and sellers.
# It provides payment infrastructure for e-commerce companies, marketplaces and individual users.
# It is expected to estimate the total transaction volume on a merchant_id and daily basis for the first 3 months of 2021.
##############
# Dataset Story
##############
# Includes data from 7 member businesses from 2018 to 2021

#4 Variant 7667 Observations 612 KB
# transaction_date Date of sales data
# merchant_id IDs of member businesses (unique number for each merchant)
# Total_Transaction Number of transactions
# Total_Paid Payment amount

#################
# Task 1: Exploration of the Dataset
#################
####################### ###
# Import Libraries
####################### ###

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
# !pip install lightgbm
# conda install lightgbm
import time
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

# Step 1: Read the Iyzico_data.csv file. Change the type of transaction_date variable to date.

DataFrame = pd.read_csv('Time_Series/datasets/iyzico_data.csv', index_col=0)
df = DataFrame.copy()

df["transaction_date"] = df["transaction_date"].apply(pd.to_datetime)

df.head()

#############################
# EDA (Exploratory Data Analysis)
#############################

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
     print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Step 2: What are the start and end dates of the data set?

df['transaction_date'].min(), df['transaction_date'].max()

# Step 3: What is the total number of transactions in each merchant?
df['merchant_id'].nunique()
df["Total_Transaction"].sum()

# Let's evaluate the sales statistics in store product breakdowns. Store performances in terms of various products.
df.groupby('merchant_id').agg({"Total_Transaction": ["sum", "mean", "median", "std"]})

# Step 4: What is the total payment amount at each merchant?
df["Total_Paid"].sum()

# Step 5: Observe the transaction count graphs of each merchant in each year.
# Graph by Total Transactions
df["Total_Transaction"].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("Total Transaction")
plt.title("İyzico Total Transaction")
x = df["transaction_date"]
plt.tight_layout()
plt.show()

# Graph by total payment
df["Total_Paid"].plot(figsize=(15, 6))
plt.xlabel("Date")
plt.ylabel("Total Paid")
plt.title("İyzico Total Paid")
x = df["transaction_date"]
plt.tight_layout()
plt.show()

# Breakdown of member businesses by total payment and total transaction
df.groupby(["merchant_id"]).agg({
     "Total_Transaction": ["sum", "mean"],
     "Total_Paid": ["sum", "mean"]})


sns.boxplot(data = df, x = df["merchant_id"], y = df["Total_Transaction"])
plt.show(block=True)

####################### ##############
# Task 2: Apply Feature Engineering techniques. Create new features.
####################### #############
# We include the US Dollar Index value on the date of the expenditures in our data set

usd = pd.read_csv('Time_Series/datasets/US_Dollar_Index.csv')
usd["Date"] = usd["Date"].apply(pd.to_datetime)
usd.columns = usd.columns.str.replace('Date', 'transaction_date')
df = df.merge(usd, how='outer', on='transaction_date')

df.head()
df.drop(["Open","High","Low","Vol.","Change %"], axis = 1, inplace = True)
df.groupby(["transaction_date"]).agg({"Total_Transaction": ["sum", "mean"],"Total_Paid": ["sum", "mean"]})

# Step 1: Date Features
# We create new variables related to dates

def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()

# Examining the number of transactions of member businesses on a yearly and monthly basis
df.groupby(["merchant_id","year","month","day_of_month"]).agg({"Total_Transaction":["sum","mean","median"]})

# Examining the total payment amounts of member businesses on a yearly and monthly basis
df.groupby(["merchant_id","year","month"]).agg({"Total_Paid":["sum","mean","median"]})


df.groupby(['merchant_id', "Total_Transaction", "month"]).agg({"Total_Paid": ["sum", "mean", "median", "std"]})


# Extensive monitoring with visualization
sns.boxplot(data = df, x = df["month"], y = df["Total_Transaction"])
plt.show(block=True)

sns.boxplot(data = df, x = df["year"], y = df["Total_Transaction"])
plt.show(block=True)

sns.boxplot(data = df, x = df["is_wknd"], y = df["Total_Transaction"])
plt.show(block=True)

sns.boxplot(data = df, x = df["day_of_month"], y = df["Total_Transaction"])
plt.show(block=True)


# Step 2: Lag/Shifted Features
# We add random noise to prevent overlearning
def random_noise(dataframe):
     return np.random.normal(scale=1.6, size=(len(dataframe),))

# Let's sort the data by date and member business number
df.sort_values(by=['merchant_id', 'transaction_date'], axis=0).head()

# We create features for different delays with sorted data.
pd.DataFrame({"Total_Transaction": df["Total_Paid"].values[0:10],
               "lag1": df["Total_Transaction"].shift(1).values[0:10],
               "lag2": df["Total_Transaction"].shift(2).values[0:10],
               "lag3": df["Total_Transaction"].shift(3).values[0:10],
               "lag4": df["Total_Transaction"].shift(4).values[0:10]})

# Let's perform this evaluation for a sale in the store product breakdown
df.groupby(['merchant_id', "Total_Transaction"])['Total_Transaction'].head()

# We automatically apply the shift method using the Lambda method with the Transform method.
df.groupby(['merchant_id', "Total_Transaction"])['Total_Transaction'].transform(lambda x: x.shift(1))

# Let's define a function. We will enter different delay values for the features calculated for the past.
def lag_features(dataframe, lags):
     for lag in lags:
         dataframe['sales_lag_' + str(lag)] = dataframe.groupby(['merchant_id', "Total_Transaction"])['Total_Transaction'].transform(
             lambda x: x.shift(lag)) + random_noise(dataframe)
     return dataframe

# It scrolls through the list of time periods and produces new features. It is dynamically named and added to the df.
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
# It will produce new features by entering various intervals and keep them for tracking.
# And noise will be added to them.

check_df(df)





# Step 3: Rolling Mean Features
# We are filling in the Na values.
# Rolling method to produce moving average features
pd.DataFrame({'Total_Transaction': df['Total_Transaction'].values[0:10],
               "roll2": df['Total_Transaction'].shift(1).rolling(window=2).mean().values[0:10],
               "roll3": df['Total_Transaction'].shift(1).rolling(window=3).mean().values[0:10],
               "roll5": df['Total_Transaction'].shift(1).rolling(window=5).mean().values[0:10]})

# We need to receive it after receiving a delay.
pd.DataFrame({"Total_Transaction": df["Total_Transaction"].values[0:10],
               "roll2": df["Total_Transaction"].shift(1).rolling(window=2).mean().values[0:10],
               "roll3": df["Total_Transaction"].shift(1).rolling(window=3).mean().values[0:10],
               "roll5": df["Total_Transaction"].shift(1).rolling(window=5).mean().values[0:10]})

# While taking the averages of the past two periods, we perform the shift operation so that the sample does not include itself.
def roll_mean_features(dataframe, windows):
     for window in windows:
         dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                           transform(
             lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
             dataframe)
     return dataframe

# Let's try to reflect moving average values and information about delays into the analysis as data features.
df = roll_mean_features(df, [91, 120, 152, 182, 242, 402, 542, 722])

# Step 4: Exponentially Weighted Mean Features

# The Roll section is adjusted as the arithmatic average of two periods, while the EWM value gives more weight to the recent period.
# Here, the roll2 variable in the first line refers to outdated variables; ewm2, on the other hand, focuses on recent data.
# We produce average features with exponential weights.
pd.DataFrame({"'Total_Transaction'": df['Total_Paid'].values[0:10],
               "roll2": df['Total_Transaction'].shift(1).rolling(window=2).mean().values[0:10],
               "ewm099": df['Total_Transaction'].shift(1).ewm(alpha=0.99).mean().values[0:10],
               "ewm095": df['Total_Transaction'].shift(1).ewm(alpha=0.95).mean().values[0:10],
               "ewm07": df['Total_Transaction'].shift(1).ewm(alpha=0.7).mean().values[0:10],
               "ewm02": df['Total_Transaction'].shift(1).ewm(alpha=0.1).mean().values[0:10]})

Let's get a # function.
def ewm_features(dataframe, alphas, lags):
     for alpha in alphas:
         for lag in lags:
             dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                 dataframe.groupby("merchant_id")['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
     return dataframe

# Let's enter the alpha and delay sets.
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 120, 152, 182, 242, 402, 542, 722]

df = ewm_features(df, alphas, lags)

# We created 72 new variables and filled in the gaps regarding delays.
# Since we have checked it with the check_df function, we can continue.

check_df(df)

# Go to the 90-day delay. When it comes to delays before the 90-day delay, focus on this 90-day delay first.
# When we give weight to past closer values, more successful predictions are made within the scope of machine learning
# We tried to reflect it in the model by creating variables.

# Step 5: Special days, exchange rate, etc.

# We can examine the properties of the created data with the grab_col_names function.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """""

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included.

    parameters
    ------
        dataframe: dataframe
                Dataframe from which variable names are to be taken
        cat_th: int, optional
                Class threshold value for variables that are numeric but categorical
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                List of cardinal variables with categorical view

    examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 lists that return equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols
    cat_cols

#######################
# Task 3: Preparation for Modeling and Modeling
#######################
# Step 1: Perform one-hot encoding.
# Since there are many variables, we will do one hot encoding using the most effective variables in order for the file to work comfortably.
# We set these variables as days of the week and year.
# Our shape value is (7691, 72) and the number of columns has increased from 67 to 77.
# (7 days a week plus 3 years variable brought 10 new columns in total)
#############################
# One-Hot Encoding
#############################

df.head()
df = pd.get_dummies(df, columns=['day_of_week','year'])
df.shape

#############################
# Converting sales to log(1+sales)
#############################

# We log the Total_Transaction variable.
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

check_df(df)

# Step 2: Define Custom Cost Functions.

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


# Step 3: Separate the data set into train and validation.
# In time series, train and validation sets; It is constructed differently from level data.
While the artificial intelligence and machine learning models we will build on # level data are expected to be divided proportionally such as 80%-20% or 75%-25%, different approaches come to the fore in time series.
# If the time series contains seasonality, the train set should end just before the period to be predicted.
# Otherwise, the model cannot apply the seasonality it has learned to the series.
# The dates we have belong to the data of 3 full years between 01.01.2018 and 31.12.2020.
# Estimates are requested for the first 3 months of 2021. We will terminate the train set just before the estimated time interval.
# We have created a train set until the beginning of 2020 (until the end of 2019).
# We need to create the first 3 months of the year as a validation set, similar to the data set we will estimate.
# In this case, we determine the first 3 months of 2020 as the validation set.

# Train set until early 2020 (end 2019).
train = df.loc[(df["transaction_date"] < "2020-01-01"), :]

train.head()

# We will be building a model with Train and verifying it with validation.

# Validation set for the first 3 months of 2020.
val = df.loc[(df["transaction_date"] >= "2020-01-01") & (df["transaction_date"] < "2020-04-01"), :]

val.head()
train.info()

# Selected all features and arguments
cols = [col for col in train.columns if col not in ['transaction_date',"Total_Transaction","Total_Paid", "year"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
# ((5105,), (5105, 78), (637,), (637, 78))

# Step 4: Create the LightGBM Model and observe the error value with SMAPE.

# !pip install lightgbm
# conda install lightgbm

# LightGBM parameters
lgb_params = {'num_leaves': 10,
               'learning_rate': 0.02,
               'feature_fraction': 0.8,
               'max_depth': 5,
               'verbose': 0,
               'num_boost_round': 10000,
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

# Since the error value did not decrease during 200 iterations, the model stopped with an early stopping warning.
# This means that the model terminates before completing the 10,000 iterations we entered into the parameters.
# If the model had not stopped, we might have faced an overfitting problem.
# Therefore, it is important to create a parameter like “early_stopping_rounds: 200”

# Estimation
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# The log was taken, we took it back and observed the errors
smape(np.expm1(y_pred_val), np.expm1(Y_val))

#24.194565216515898

# If we set the time of data as 28/8 months as train and validation; we are going to get SMAPE: 32.93507139156651 (residual score)
# So that is why we set the data with seasonality.



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

# The sales_roll_mean91 value, which shows a 3-month delay, appears as the most important explanatory variable.
# Immediately afterwards, variables regarding the delays of the 4th and 5th months provide high explanatory value.


#------------------------------------------------- ----------------------------

feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

#------------------------------------------------- ----------------------------

#############################
# Final Model
#############################
# We continue by adding the 9-month lost time in the data set to the prediction file and creating the final model.

cols = [col for col in df.columns if col not in ["Total_Transaction","transaction_date"]]
Y_train = df['Total_Transaction']
X_train = df[cols]

Y_train.shape,X_train.shape

lgb_params = {'num_leaves': 20,
               'learning_rate': 0.02,
               'feature_fraction': 0.9,
               'max_depth': 15,
               'verbose': 0,
               'num_boost_round': model.best_iteration,
               'nthread':-1}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_val, predict_disable_shape_check=True, num_iteration=model.best_iteration)
# The logarithmed values are the test_preds values above.

#############################
# Submission File
#############################
df.head()
val.head()

# merchant_id and Total_Paid for the test set
submission_df = val.loc[:, ["merchant_id", "Total_Paid"]]

# Let's place the values we estimated ourselves (we take the log inverse) for the 90-day period
submission_df['Total_Paid'] = np.expm1(test_preds)

# We define Merchant_id id.
submission_df['merchant_id'] = submission_df['merchant_id'].astype(int)

# We are doing the external registration process.
submission_df.to_csv("submission_demand.csv", index=False)