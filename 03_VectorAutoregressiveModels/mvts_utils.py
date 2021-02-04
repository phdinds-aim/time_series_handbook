import numpy as np
import pandas as pd
import itertools
import statsmodels.tsa as tsa
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, adfuller, ccf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def fit_arima(train, 
              p_list=[1,2,3,4], 
              d_list=[1],
              q_list=[1,2,3,4]):
    aic, bic, hqic = [], [], []
    pdqs = list(itertools.product(p_list, d_list, q_list))
    index=[]
    for pdq in pdqs:
        try:
            model = ARIMA(train, order=pdq)
            result = model.fit()
            aic.append(result.aic)
            bic.append(result.bic)
            hqic.append(result.hqic)
            index.append(pdq)
        except ValueError:
            continue
            
    order_metrics_df = pd.DataFrame({'AIC': aic, 
                                     'BIC': bic, 
                                     'HQIC': hqic}, 
                                    index=index)   
        
    return order_metrics_df

def forecast_arima(train, test, order):
    history = list(train)
    predictions = list()
    model = ARIMA(history, order=order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=len(test))
    predictions = output[0]
    return np.array(predictions).flatten()


# def forecast_naive(train, test):
#     lr = LinearRegression()
#     x_train = train[:-1]
#     y_train = train[1:]
#     lr.fit(x_train.reshape(-1, 1), y_train)
#     forecast_lr = [lr.predict(np.array([train[-1]]).reshape(-1, 1))]
#     for n in range(len(test)-1):
#         forecast_lr.append(lr.predict(np.array([forecast_lr[-1]]).reshape(-1, 1)))
#     forecast_lr = np.hstack(forecast_lr)
#     return forecast_lr

def cross_corr_mat(df, yi_col, yj_col, lag=0):
    yi_yi = acf(df[yi_col].values, unbiased=False, nlags=len(df)-2)
    yj_yj = acf(df[yj_col].values, unbiased=False, nlags=len(df)-2)
    yi_yj = ccf(df[yi_col].values, df[yj_col].values, unbiased=False)
    yj_yi = ccf(df[yj_col].values, df[yi_col].values, unbiased=False)
    ccm = pd.DataFrame({yi_col: [yi_yi[lag], yj_yi[lag]],
                        yj_col: [yi_yj[lag], yj_yj[lag]]}, 
                       index=[yi_col, yj_col])
    return ccm

def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'-d1'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'-d2'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'-forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'-d1'].cumsum()
    return df_fc



def plot_forecasts_static(train_df, 
                          test_df, 
                          forecast_df, 
                          column_name,
                          min_train_date=None, 
                          title='',
                          suffix=['-forecast']):
    train_df = pd.concat([train_df, test_df.iloc[:1]])
    if min_train_date is not None: 
        train_df = train_df.loc[train_df.index>=min_train_date]
    fig, ax = plt.subplots(figsize=(16, 2.5), sharex=True)
    train_df[column_name].plot(ax=ax)
    test_df[column_name].plot(ax=ax)
    for s in suffix:
        forecast_df[column_name+s].plot(ax=ax)
    plt.legend(['train', 'test'] + [s.split('-')[-1] for s in suffix], loc=2)
    plt.title(title)
    plt.tight_layout()
    return fig, ax

def plot_forecasts_interactive(train_df,
                               test_df, 
                               forecast_df, 
                               column_name, 
                               suffix='-forecast'):
    fig = go.Figure()
    train_df = pd.concat([train_df, test_df.iloc[:1]])
    fig.add_trace(
        go.Scatter(name="train", 
                x=list(train_df.index), 
                y=list(train_df[column_name])))
    fig.add_trace(
        go.Scatter(name="test", 
                x=list(test_df.index),
                y=list(test_df[column_name])))
    fig.add_trace(
        go.Scatter(name='VAR forecast',
                x=list(forecast_df.index), 
                y=list(forecast_df[column_name+suffix])))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=250,
        margin=dict(
            l=60,
            r=60,
            b=30,
            t=30,
        )
    )
    return fig

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def test_performance_metrics(test_df, forecast_df, suffix='-VAR'):
    mae = []
    mse = []
    mape = []
    cols = test_df.columns
    for c in cols:
        mae.append(mean_absolute_error(test_df[c], forecast_df[c+suffix]))
        mse.append(mean_squared_error(test_df[c], forecast_df[c+suffix]))
        mape.append(mean_absolute_percentage_error(test_df[c], forecast_df[c+suffix]))
    metrics_df = pd.DataFrame({'MAE': mae,
                               'MSE': mse, 
                               'MAPE': mape}, index=[c+suffix for c in cols])
    return metrics_df.T