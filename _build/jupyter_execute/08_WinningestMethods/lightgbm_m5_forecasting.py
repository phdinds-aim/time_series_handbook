# Chapter 8: Winningest Methods in Time Series Forecasting

Compiled by: Sebastian C. Ibañez

In previous sections, we examined several models used in time series forecasting such as ARIMA, VAR, and Exponential Smoothing methods. While the main advantage of traditional statistical methods is their ability to perform more sophisticated inference tasks directly (e.g. hypothesis testing on parameters, causality testing), they usually lack predictive power because of their rigid assumptions. That is not to say that they are <i>necessarily</i> inferior when it comes to forecasting, but rather they are typically used as performance benchmarks.

In this section, we demonstrate several of the fundamental ideas and approaches used in the recently concluded [`M5 Competition`](https://mofc.unic.ac.cy/m5-competition/) where challengers from all over the world competed in building time series forecasting models for both [`accuracy`](https://www.kaggle.com/c/m5-forecasting-accuracy) and [`uncertainty`](https://www.kaggle.com/c/m5-forecasting-uncertainty) prediction tasks. Specifically, we explore the machine learning model that majority of the competition's winners utilized: [`LightGBM`](https://lightgbm.readthedocs.io/en/latest/index.html), a tree-based gradient boosting framework designed for speed and efficiency.

## 1. M5 Dataset

You can download the M5 dataset from the Kaggle links above. 

Let's load the dataset and examine it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_x_size = 15
plot_y_size = 2

np.set_printoptions(precision = 6, suppress = True)

date_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(start = '2011-01-29', end = '2016-04-24')]

df_calendar = pd.read_csv('../data/m5/calendar.csv')
df_price = pd.read_csv('../data/m5/sell_prices.csv')
df_sales = pd.read_csv('../data/m5/sales_train_validation.csv')

df_sales.rename(columns = dict(zip(df_sales.columns[6:], date_list)), inplace = True)
df_sales

df_calendar

df_price

#### Sample Product

Let's choose a random product and plot it.

df_sample = df_sales.iloc[3, :]
series_sample = df_sample.iloc[6:]

df_sample

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series_sample.plot()
plt.legend()
plt.show()

#### Pick a Time Series

Let's try and find an interesting time series to forecast.

df_sales_total_by_store = df_sales.groupby(['store_id']).sum()
df_sales_total_by_store

plt.rcParams['figure.figsize'] = [plot_x_size, 4]

df_sales_total_by_store.T.plot()
plt.show()

series = df_sales_total_by_store.iloc[0]
print(series.name)
print('Min Dates:' + str(series[series == series.min()].index.to_list()))

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series.plot()
plt.legend()
plt.show()

## 2. Pre-processing

Before we build a forecasting model, let's check some properties of our time series.

### Is the series non-stationary?

Let's check.

from statsmodels.tsa.stattools import adfuller

result = adfuller(series)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

### Does differencing make the series stationary?

Let's check.

def difference(dataset, interval = 1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
 
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

series_d1 = difference(series)
result = adfuller(series_d1)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

### Is the series seasonal?

Let's check.

from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

plot_acf(series)
plt.show()

plot_acf(series, lags = 730, use_vlines = True)
plt.show()

### Can we remove the seasonality?

Let's check.

series_d7 = difference(series, 7)

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

plot_acf(series_d7)
plt.show()

plot_acf(series_d7, lags = 730, use_vlines = True)
plt.show()

series_d7_d30 = difference(series_d7, 30)

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

plot_acf(series_d7_d30)
plt.show()

plot_acf(series_d7_d30, lags = 730, use_vlines = True)
plt.show()

result = adfuller(series_d7_d30)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

series_d7_d30 = pd.Series(series_d7_d30)
series_d7_d30.index = date_list[37:]

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series_d7_d30.plot(label = 'Differenced Series')
plt.legend()
plt.show()

### What now?

At this point we have two options:

- Model the seasonally differenced series, then reverse the differencing after making predictions.

- Model the original series directly.

While (vanilla) ARIMA requires a non-stationary and non-seasonal time series, these properties are not necessary for most non-parametric ML models.

## 3. One-Step Prediction

Let's build a model for making one-step forecasts.

To do this, we first need to transform the time series data into a supervised learning dataset.

In other words, we need to create a new dataset consisting of $X$ and $Y$ variables, where $X$ refers to the features and $Y$ refers to the target.

### How far do we lookback?

To create the new $(X,Y)$ dataset, we first need to decide what the $X$ features are. 

For the moment, let's ignore any exogenous variables. In this case, what determines the $X$s is how far we <i>lookback</i>. In general, we can treat the lookback as a hyperparameter, which we will call `window_size`.

<i>Advanced note:</i> Technically, we could build an entire methodology for feature engineering $X$.

### Test Set

To test our model we will use the last 28 days of the series.

### CREATE X,Y ####
def create_xy(series, window_size, prediction_horizon, shuffle = False):
    x = []
    y = []
    for i in range(0, len(series)):
        if len(series[(i + window_size):(i + window_size + prediction_horizon)]) < prediction_horizon:
            break
        x.append(series[i:(i + window_size)])
        y.append(series[(i + window_size):(i + window_size + prediction_horizon)])
    x = np.array(x)
    y = np.array(y)
    return x,y

### HYPERPARAMETERS ###
window_size = 365
prediction_horizon = 1

### TRAIN VAL SPLIT ### (include shuffling later)
test_size = 28
split_time = len(series) - test_size

train_series = series[:split_time]
test_series = series[split_time - window_size:]

train_x, train_y = create_xy(train_series, window_size, prediction_horizon)
test_x, test_y = create_xy(test_series, window_size, prediction_horizon)

train_y = train_y.flatten()
test_y = test_y.flatten()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series[-test_size:].plot(label = 'CA_1 Test Series')
plt.legend()
plt.show()

###  LightGBM

Now we can build a LightGBM model to forecast our time series.

Gradient boosting is an ensemble method that combines multiple weak models to produce a single strong prediction model. The method involves constructing the model (called a <i>gradient boosting machine</i>) in a serial stage-wise manner by sequentially optimizing a differentiable loss function at each stage. Much like other boosting algorithms, the residual errors are passed to the next weak learner and trained.

For this work, we use LightGBM, a gradient boosting framework designed for speed and efficiency. Specifically, the framework uses tree-based learning algorithms.

To tune the model's hyperparameters, we use a combination of grid search and repeated k-fold cross validation, with some manual tuning. For more details, see the Hyperparameter Tuning notebook.

Now we train the model on the full dataset and test it.

import lightgbm as lgb

params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'num_leaves': 2**4,
    'learning_rate': 0.1,
    'boosting_type': 'dart'
}

model = lgb.LGBMRegressor(first_metric_only = True, **params)

model.fit(train_x, train_y,
          eval_metric = 'l1', 
          eval_set = [(test_x, test_y)],
          #early_stopping_rounds = 10,
          verbose = 0)

forecast = model.predict(test_x)
s1_naive = series[-29:-1].to_numpy()
s7_naive = series[-35:-7].to_numpy()
s30_naive = series[-56:-28].to_numpy()
s365_naive = series[-364:-336].to_numpy()

print('     Naive MAE: %.4f' % (np.mean(np.abs(s1_naive - test_y))))
print('  s7-Naive MAE: %.4f' % (np.mean(np.abs(s7_naive - test_y))))
print(' s30-Naive MAE: %.4f' % (np.mean(np.abs(s30_naive - test_y))))
print('s365-Naive MAE: %.4f' % (np.mean(np.abs(s365_naive - test_y))))
print('  LightGBM MAE: %.4f' % (np.mean(np.abs(forecast - test_y))))

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series[-test_size:].plot(label = 'True')
plt.plot(forecast, label = 'Forecast')
plt.legend()
plt.show()

### Tuning Window Size
How does our metric change as we extend the window size?

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'num_leaves': 2**4,
    'learning_rate': 0.1,
    'boosting_type': 'dart'
}

windows = [7, 30, 180, 365, 545, 730]

results = []
names = []
for w in windows:

    window_size = w

    train_x, train_y = create_xy(train_series, window_size, prediction_horizon)

    train_y = train_y.flatten()

    cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 123)
    scores = cross_val_score(lgb.LGBMRegressor(**params), train_x, train_y, scoring = 'neg_mean_absolute_error', cv = cv, n_jobs = -1)
    results.append(scores)
    names.append(w)
    print('%3d --- MAE: %.3f (%.3f)' % (w, np.mean(scores), np.std(scores)))

plt.rcParams['figure.figsize'] = [plot_x_size, 5]    
    
plt.boxplot(results, labels = names, showmeans = True)
plt.show()

## 4. Multi-Step Prediction

Suppose we were interested in forecasting the next $n$-days instead of just the next day.

There are several approaches we can take to solve this problem.

### HYPERPARAMETERS ###
window_size = 365
prediction_horizon = 1

### TRAIN VAL SPLIT ###
test_size = 28
split_time = len(series) - test_size

train_series = series[:split_time]
test_series = series[split_time - window_size:]

train_x, train_y = create_xy(train_series, window_size, prediction_horizon)
test_x, test_y = create_xy(test_series, window_size, prediction_horizon)

train_y = train_y.flatten()
test_y = test_y.flatten()

### Recursive Forecasting

In recursive forecasting, we first train a one-step model then generate a multi-step forecast by recursively feeding our predictions back into the model.

params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'num_leaves': 2**4,
    'learning_rate': 0.1,
    'boosting_type': 'dart'
}

model = lgb.LGBMRegressor(first_metric_only = True, **params)

model.fit(train_x, train_y,
          eval_metric = 'l1', 
          eval_set = [(test_x, test_y)],
          #early_stopping_rounds = 10,
          verbose = 0)

recursive_x = test_x[0, :]

forecast_ms = []
for i in range(test_x.shape[0]):
    pred = model.predict(recursive_x.reshape((1, recursive_x.shape[0])))
    recursive_x = np.append(recursive_x[1:], pred)
    forecast_ms.append(pred)
    
forecast_ms_rec = np.asarray(forecast_ms).flatten()
forecast_os = model.predict(test_x)

print('  One-Step MAE: %.4f' % (np.mean(np.abs(forecast_os - test_y))))
print('Multi-Step MAE: %.4f' % (np.mean(np.abs(forecast_ms_rec - test_y))))

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series[-test_size:].plot(label = 'True')
plt.plot(forecast_ms_rec, label = 'Forecast Multi-Step')
plt.plot(forecast_os, label = 'Forecast One-Step')
plt.legend()
plt.show()

### Direct Forecasting

In direct forecasting, we train $n$ independent models and generate a multi-step forecast by concatenating the $n$ predictions.

For this implementation, we need to create a new $(X,Y)$ dataset, where $Y$ is now a vector of $n$ values.

### HYPERPARAMETERS ###
window_size = 365
prediction_horizon = 28

### TRAIN VAL SPLIT ###
test_size = 28
split_time = len(series) - test_size

train_series = series[:split_time]
test_series = series[split_time - window_size:]

train_x, train_y = create_xy(train_series, window_size, prediction_horizon)
test_x, test_y = create_xy(test_series, window_size, prediction_horizon)

from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(lgb.LGBMRegressor(), n_jobs = -1)

model.fit(train_x, train_y)

forecast_ms_dir = model.predict(test_x)

print('  One-Step MAE: %.4f' % (np.mean(np.abs(forecast_os - test_y))))
print('Multi-Step MAE: %.4f' % (np.mean(np.abs(forecast_ms_dir - test_y))))

plt.rcParams['figure.figsize'] = [plot_x_size, plot_y_size]

series[-test_size:].plot(label = 'True')
plt.plot(forecast_ms_dir.T, label = 'Forecast Multi-Step')
plt.plot(forecast_os, label = 'Forecast One-Step')
plt.legend()
plt.show()

### Single-Shot Forecasting

In single-shot forecasting, we create a model that attempts to predict all $n$-steps simultaneously.

Unfortunately, LightGBM (tree-based methods in general) does not support multi-output models.

### Forecast Combination

An easy way to improve forecast accuracy is to use several different methods on the same time series, and to average the resulting forecasts.

forecast_ms_comb = 0.5*forecast_ms_dir.flatten() + 0.5*forecast_ms_rec

print('  Recursive MAE: %.4f' % (np.mean(np.abs(forecast_ms_rec - test_y))))
print('     Direct MAE: %.4f' % (np.mean(np.abs(forecast_ms_dir - test_y))))
print('Combination MAE: %.4f' % (np.mean(np.abs(forecast_ms_comb - test_y))))

series[-test_size:].plot(label = 'True')
plt.plot(forecast_ms_comb, label = 'Forecast Combination')
plt.show()

## 5. Feature Importance

One advantage of GBM models is that it can generate feature importance metrics based on the quality of the splits (or information gain).

### HYPERPARAMETERS ###
window_size = 365
prediction_horizon = 1

### TRAIN VAL SPLIT ###
test_size = 28
split_time = len(series) - test_size

train_series = series[:split_time]
test_series = series[split_time - window_size:]

train_x, train_y = create_xy(train_series, window_size, prediction_horizon)
test_x, test_y = create_xy(test_series, window_size, prediction_horizon)

train_y = train_y.flatten()
test_y = test_y.flatten()
    
params = {
    'n_estimators': 2000,
    'max_depth': 4,
    'num_leaves': 2**4,
    'learning_rate': 0.1,
    'boosting_type': 'dart'
}

model = lgb.LGBMRegressor(first_metric_only = True, **params)

feature_name_list = ['lag_' + str(i+1) for i in range(window_size)]

model.fit(train_x, train_y,
          eval_metric = 'l1', 
          eval_set = [(test_x, test_y)],
          #early_stopping_rounds = 10,
          feature_name = feature_name_list,
          verbose = 0)

plt.rcParams['figure.figsize'] = [5, 5]

lgb.plot_importance(model, max_num_features = 15, importance_type = 'split')
plt.show()

## Summary

In summary, this section has shown the following:

- A quick exploration of the M5 dataset and its salient features
- Checking the usual assumptions for classical time series analysis
- Demonstrating the 'out-of-the-box' performance of LightGBM models
- Demonstrating how tuning the lookback window can effect forecasting performance
- Demonstrating how tuning the hyperparameters of a LightGBM model can improve performance
- Summarizing the different approaches to multi-step forecasting
- Illustrating that gradient boosting methods can measure feature importance

Many of the methods and approaches shown above are easily extendable. Of note, because of the non-parametric nature of most machine learning models, performing classical inference tasks like hypothesis testing is more challenging. However, trends in ML research have been moving towards greater interpretability of so-called 'black box' models, such as SHAP for feature importance and even Bayesian approaches to neural networks for causality inference.

## References

[1] S. Makridakis, E. Spiliotis, and V. Assimakopoulos. The M5 Accuracy competition: Results, findings and conlusions. 2020.

[2] S. Makridakis, E. Spiliotis, V. Assimakopoulos, Z. Chen, A. Gaba, I. Tsetlin, and R. Winkler. The M5 Uncertainty competition: Results, findings and conlusions. 2020.

[3] V. Jose, and R. Winkler. Evaluating quantile assessments. Operations research, 2009.

[4] A. Gaba, I. Tsetlin, and R. Winkler. Combining interval forecasts. Decision Analysis, 2017.




```{toctree}
:hidden:
:titlesonly:


lightgbm_m5_tuning
lightgbm_jena_forecasting
```
