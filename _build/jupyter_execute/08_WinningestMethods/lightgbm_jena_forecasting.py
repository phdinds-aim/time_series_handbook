# Temperature Forecasting (Supplementary Notebook)

Similar to what was done in previous sections, this notebook applies the methodology used in the M5 Forecasting notebook to the Jena Climate dataset. Specifically, we will be forecasting the temperature variable.

## 1. Jena Climate Dataset

Let's load the dataset and examine it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.rcParams['figure.figsize'] = [15, 5]

np.set_printoptions(precision = 6, suppress = True)

df = pd.read_csv('../data/jena_climate_2009_2016.csv')
df.shape

df.head(10)

series = df['T (degC)'].iloc[5:]

series.plot()
plt.show()
print(series.shape)

series = series.iloc[::6]

series.plot()
plt.show()
print(series.shape)

series = df.iloc[::6, 1:]
series

### HYPERPARAMETERS ###
window_size = 240
prediction_horizon = 1

### TRAIN TEST VAL SPLIT ###
train_series = series.iloc[:35045, 1]
val_series = series.iloc[35045:52569, 1]
test_series = series.iloc[52569:, 1]

train_x, train_y = create_xy(train_series.to_numpy(), window_size, prediction_horizon)
val_x, val_y = create_xy(val_series.to_numpy(), window_size, prediction_horizon)
test_x, test_y = create_xy(test_series.to_numpy(), window_size, prediction_horizon)

train_y = train_y.flatten()
val_y = val_y.flatten()
test_y = test_y.flatten()

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
print(test_x.shape)
print(test_y.shape)

test_series.plot()
plt.show()

###  LightGBM

import lightgbm as lgb

model = lgb.LGBMRegressor()

model.fit(train_x, train_y,
          eval_metric = 'l1', 
          eval_set = [(val_x, val_y)],
          early_stopping_rounds = 100,
          verbose = 0)

forecast = model.predict(test_x)
print('  LightGBM MAE: %.4f' % (np.mean(np.abs(forecast - test_y))))

#series[-test_size:].plot(marker = 'o', linestyle = '--')
#plt.plot(forecast, marker = 'o', linestyle = '--')
#plt.show()

### HYPERPARAMETERS ###
window_size = 240
prediction_horizon = 24

### TRAIN TEST VAL SPLIT ###
train_series = series.iloc[:35045, 1]
val_series = series.iloc[35045:52569, 1]
test_series = series.iloc[52569:, 1]

train_x, train_y = create_xy(train_series.to_numpy(), window_size, prediction_horizon)
val_x, val_y = create_xy(val_series.to_numpy(), window_size, prediction_horizon)
test_x, test_y = create_xy(test_series.to_numpy(), window_size, prediction_horizon)

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
print(test_x.shape)
print(test_y.shape)

### LightGBM

recursive_x = test_x
forecast_ms = []
for j in range(prediction_horizon):
    pred = model.predict(recursive_x)
    recursive_x = np.hstack((recursive_x[:, 1:], pred[:, np.newaxis]))
    forecast_ms.append(pred)
forecast_ms = np.asarray(forecast_ms).T
print('LightGBM')
print('MAE:', np.mean(np.abs(test_y - forecast_ms)))
print('RMSE:', np.sqrt(np.mean((test_y - forecast_ms)**2)))
print('')

### Naive

tmp = test_y[:-1, -1]
n_forecast = []

for val in tmp:
    n_forecast.append(np.repeat(val, 24))
n_forecast = np.asarray(n_forecast)

print('Naive')
print('MAE:', np.mean(np.abs(test_y[1:, :] - n_forecast)))
print('RMSE:', np.sqrt(np.mean((test_y[1:, :] - n_forecast)**2)))
print('')


### S.Naive

new_test = np.vstack(np.array_split(test_series.to_numpy()[3:], 730))

print('Seasonal Naive')
print('MAE:', np.mean(np.abs(new_test[1:, :] - new_test[:-1, :])))
print('RMSE:', np.sqrt(np.mean((new_test[1:, :] - new_test[:-1, :])**2)))

## Summary

Using LightGBM with a recursive forecasting strategy demonstrates that we can achieve an MAE of about 2.08°C without hyperparameter tuning.