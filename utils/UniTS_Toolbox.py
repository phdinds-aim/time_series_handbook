# Load modules
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from math import sqrt
from pandas import Series
import matplotlib.pyplot as plt
import plotly.express as px
import pylab as plot
import plotly.offline as pyoff
import plotly.tools as tools
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.api as smtsa
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Constant Helpers

# For Plotting
STYLE = 'ggplot'
plt.style.use(STYLE)
FIGSIZE = (14, 7)
FIGDPI = 180

# Taken from Prof. Felix's Univariate Time Series Notebook
REGRESSION = "c"
AUTOLAG = "AIC"
NLAGS = "auto"

# Differencing Names
ORIGINAL_NAME = 'Original'
FIRST_DIFF_NAME = '1st Order'
SECOND_DIFF_NAME = '2nd Order'

# “Life, the universe, and everything”!
SEED = 42

# For Modeling
SPLIT = 0.80

# Path of the dataset
DATA_PATH = r"../data/household_power_consumption.txt.zip"

class UniTS_Toolbox:

    def __init__(self, data=None, lags=None):
        """Initialize the toolbox

        Parameters
        ----------
        data {Series}
            - dataset
        lags {int, array_like}, optional
            - Lag values that can be an int or array
        figsize (float, float)
            - Width, height in inches
        """
        self.data = data
        self.lags = lags

    # Function for preprocessing
    def preprocess_dataset(self, resolution, all_cols=False):
        """Perform preprocessing of the dataset
        Parameters
        ----------
        resolution (string)
        - resolution (e.g., "M" - Monthly, "W" - Weekly, "D" - Daily)

        Output
        ----------
        Time series data 
        """
        with zipfile.ZipFile(DATA_PATH) as power_z:
            with power_z.open("household_power_consumption.txt") as power_data:
                # read the dataset from ./data/household_power_consumption.txt.zip
                df = pd.read_csv(power_data,
                                 sep=';',
                                 parse_dates={'dt': ['Date', 'Time']},
                                 infer_datetime_format=True,
                                 low_memory=False,
                                 na_values=['nan', '?'],
                                 index_col='dt')
        df.columns = [x.lower() for x in df.columns]

        # features
        relevant_columns = [
            "global_active_power", "global_reactive_power", "voltage",
            "global_intensity", "sub_metering_1", "sub_metering_2",
            "sub_metering_3"
        ]

        # Handling missing values
        for every_column in relevant_columns:
            df[every_column] = df[every_column].interpolate()

        # Computation for overall power consumtion
        eq1 = (df['global_active_power'] * 1000 / 60)
        eq2 = df['sub_metering_1'] + \
            df['sub_metering_2'] + df['sub_metering_3']
        df['power_consumption'] = eq1 - eq2
        relevant_columns = df.columns

        # Aggregating level: "M" - Monthly, "W" - Weekly, "D" - Daily
        df = df[relevant_columns].resample(resolution).sum()
        if not all_cols:
            df_power_consumption = df['power_consumption']
        else:
            df_power_consumption = df
        print("DONE: Data Preprocessing")
        return df_power_consumption

    # Function for EDA
    def plot_it(self,
                x1=None,
                y1=None,
                x2=None,
                y2=None,
                x3=None,
                y3=None,
                x_title=None,
                y_title=None,
                title=None,
                t1_name=None,
                t2_name=None,
                t3_name=None,
                mode=None):
        """Plot Graphs using Plotly

        Parameters
        ----------
        x1 {float}
            - trace 1 numbers to be plotted in x
        y1 {float}
            - trace 1 numbers to be plotted in y
        x2 {float}
            - trace 2 numbers to be plotted in x
        y2 {float}
            - trace 2 numbers to be plotted in y
        x3 {float}
            - trace 3 numbers to be plotted in x
        y3 {float}
            - trace 3 numbers to be plotted in y
        x_title {string}(default: None)
            - label in x-axis
        y_title {string}(default: None)
            - label in y-axis
        title (string)(default: None)
            - plot title
        t1_name (string)(default: None)
            - trace 1 title
        t2_name (string)(default: None)
            - trace 2 title
        t3_name (string)(default: None)
            - trace 3 title
        mode (string)(default: None)
            - type of the marker
        """
        trace1 = go.Scatter(x=x1,
                            y=y1,
                            mode=mode,
                            marker=dict(size=12,
                                        color='royalblue',
                                        line=dict(width=3,
                                                  color='rgba(0, 0, 0, 1.0)')),
                            name=t1_name)
        trace2 = go.Scatter(x=x2,
                            y=y2,
                            mode=mode,
                            marker=dict(size=12,
                                        color='firebrick',
                                        line=dict(width=3,
                                                  color='rgba(0, 0, 0, 1.0)')),
                            name=t2_name)
        trace3 = go.Scatter(x=x3,
                            y=y3,
                            mode=mode,
                            marker=dict(size=12,
                                        color='green',
                                        line=dict(width=3,
                                                  color='rgba(0, 0, 0, 1.0)')),
                            name=t3_name)
        layout = go.Layout(xaxis=dict(showgrid=True,
                                      title=x_title,
                                      titlefont=dict(size=15),
                                      tickfont=dict(size=15)),
                           yaxis=dict(showgrid=True,
                                      title=y_title,
                                      titlefont=dict(size=15),
                                      tickfont=dict(size=15)),
                           title=title,
                           titlefont=dict(size=20),
                           template='ggplot2')
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        pyoff.iplot(fig)

    # Function for generating Time Series Data
    def gen_arma_data(self, ar, ma, seed=SEED, sample=500):
        """Generate Time Series Data

        Parameters
        ----------
        ar (list)
            - The coefficient for autoregressive lag polynomial, including zero lag
        ma (list)
            - The coefficient for autoregressive lag polynomial, including zero lag 
        sample(int)   
            - number of samples

        Return
        ----------
        data - Random sample(s) from an ARMA process.
        """
        np.random.seed(SEED)
        arparams = np.array(ar)
        maparams = np.array(ma)
        ar = np.r_[1, -arparams]
        ma = np.r_[1, maparams]
        data = smtsa.arma_generate_sample(ar=ar, ma=ma, nsample=sample)
        return (data)

    # Function for plotting and testing Time Series using ADF and KPSS
    def plot_test(self, figsize=FIGSIZE):
        """Plot and Test Time Series

        Parameters
        ----------
        figsize (float, float)
            - Width, height in inches
        """
        if not isinstance(self.data, pd.Series):
            self.data = pd.Series(self.data)
        with plt.style.context(style=STYLE):
            fig = plt.figure(figsize=FIGSIZE)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            self.data.plot(ax=ts_ax)
            adf_p_value = sm.tsa.stattools.adfuller(self.data)[1]
            kpss_p_value = sm.tsa.stattools.kpss(self.data)[1]
            ts_ax.set_title(
                'Time Series Analysis Plots\n Tests for Stationary\n ADF: p={0:.2f}\n KPSS: p={1:.2f}'
                .format(adf_p_value, kpss_p_value))
            plot_acf(self.data, lags=self.lags, ax=acf_ax)
            plot_pacf(self.data, lags=self.lags, ax=pacf_ax)

            # Perform Stationarity Test ADF & KPSS tests
            self.adf_test(self.data.dropna(),
                          regression=REGRESSION,
                          autolag=AUTOLAG)
            self.kpss_test(self.data.dropna(),
                           regression=REGRESSION,
                           nlags=NLAGS)
            plt.tight_layout()

    # Function for ADF Test
    def adf_test(self, timeseries, *args, **kwargs):
        # Convenience functions for ADF and KPSS tests from statsmodels.org
        print('\nResults of Dickey-Fuller Test:')
        adf_test = adfuller(timeseries, *args, **kwargs)
        adf_test_out = pd.Series(adf_test[0:4],
                                 index=[
                                     'Test Statistic', 'p-value', '#Lags Used',
                                     'Number of Observations Used'
                                 ])
        for key, value in adf_test[4].items():
            adf_test_out['Critical Value (%s)' % key] = value

        # Added for interpretation

        # using Test Statistic
        if adf_test_out['Critical Value (1%)'] < adf_test_out['Test Statistic']:
            print(
                'Output(using Test Statistic): NON-Stationary with 99% confidence. '
            )
        elif adf_test_out['Critical Value (5%)'] < adf_test_out[
                'Test Statistic']:
            print(
                'Output(using Test Statistic): NON-Stationary with 95% confidence. '
            )
        elif adf_test_out['Critical Value (10%)'] < adf_test_out[
                'Test Statistic']:
            print(
                'Output(using Test Statistic): NON-Stationary with 90% confidence. '
            )
        else:
            print('Output(using Test Statistic): STATIONARY! ')
        # using p-value
        p_value = adf_test_out['p-value']
        print(
            f'Output(using p-value): {"NOT " if p_value > 0.05 else ""}STATIONARY!'
        )
        print(adf_test_out)

    # Function for KPSS Test
    def kpss_test(self, timeseries, *args, **kwargs):
        # Convenience functions for ADF and KPSS tests, taken from statsmodels.org
        print('\nResults of KPSS Test:')
        kpss_test = kpss(timeseries, *args, **kwargs)
        kpss_test_out = pd.Series(
            kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
        for key, value in kpss_test[3].items():
            kpss_test_out['Critical Value (%s)' % key] = value

        # Added for interpretation

        # using Test Statistic
        if abs(kpss_test_out['Critical Value (1%)']) < abs(
                kpss_test_out['Test Statistic']):
            print(
                'Output(using Test Statistic): NON-Stationary with 99% confidence. '
            )
        elif abs(kpss_test_out['Critical Value (5%)']) < abs(
                kpss_test_out['Test Statistic']):
            print(
                'Output(using Test Statistic): NON-Stationary with 95% confidence. '
            )
        elif abs(kpss_test_out['Critical Value (10%)']) < abs(
                kpss_test_out['Test Statistic']):
            print(
                'Output(using Test Statistic): NON-Stationary with 90% confidence. '
            )
        else:
            print('Output(using Test Statistic): STATIONARY!')

        # using p-value
        p_value = kpss_test_out['p-value']
        print(
            f'Output(using p-value): {"NOT " if p_value <= 0.05 else ""}STATIONARY!'
        )
        print(kpss_test_out)

    # Function for plotting ARIMA & SARIMA
    def plot_s_arima(self, train, test, forecast, title):
        """Plot ARIMA Series

        Parameters
        ----------
        train, test, forecast {Series}
            - train, test, forecast series data
        """
        plt.figure(figsize=FIGSIZE)
        plt.plot(train, color='red', label='Train')
        plt.plot(test, color='green', label='Test')
        plt.plot(forecast, color='blue', label='Forecast')
        plt.title(title)
        plt.xlabel('Year-Month')
        plt.ylabel('Power Consumption')
        plt.legend(loc="best")
        plt.show()

    # Function for differencing
    def differencing(self, data, iteration=1):
        """Perform Manual Differencing
        Parameters
        ----------
        data {Series}
            - dataset
        iteration {integer}
            - number of interval

        Return
        ----------
        diff_list - Differenced Series
        """
        diff_list = list()
        for i in range(iteration, len(data)):
            value = data[i] - data[i - iteration]
            diff_list.append(value)
        return Series(diff_list)

    # Function for stationarity test
    def stationarity_test(self,
                          original=None,
                          first_diff=None,
                          second_diff=None,
                          method=None):
        """Perform stationarity test, differencing up to 2nd, and plot the results
        Parameters
        ----------
        Original {Series}
            - Original
        first_diff {Series}
            - 1st Order difference
        second_diff {Series}
            - 2nd Order difference
        method {string}
            - Differencing Method
        """
        plt.rcParams.update({'figure.figsize': FIGSIZE, 'figure.dpi': FIGDPI})
        fig, axes = plt.subplots(3, 3)
        for t in range(0, 3):
            if t == 0:
                diff_var = original
                diff_name = ORIGINAL_NAME
            elif t == 1:
                diff_var = first_diff
                diff_name = FIRST_DIFF_NAME
            elif t == 2:
                diff_var = second_diff
                diff_name = SECOND_DIFF_NAME
            else:
                None

            # Plot the time series
            axes[t, 0].plot(diff_var)
            axes[t, 0].set_title('(%s) Series' % (diff_name))

            # Plot Autocorrelation
            plot_acf(diff_var.dropna(),
                     lags=None,
                     ax=axes[t, 1],
                     title='Autocorrelation for (%s)' % diff_name)

            # Plot Partial Autocorrelation
            plot_pacf(diff_var.dropna(),
                      lags=None,
                      ax=axes[t, 2],
                      title='Partial Autocorrelation for (%s)' % diff_name)
            print('\nStatistic for (%s)' % (diff_name))

            # Perform Stationarity Test ADF & KPSS tests
            self.adf_test(diff_var.dropna(),
                          regression=REGRESSION,
                          autolag=AUTOLAG)
            self.kpss_test(diff_var.dropna(),
                           regression=REGRESSION,
                           nlags=NLAGS)
        fig.suptitle('Plots and Statistics for (%s)' % method, fontsize=20)
        plt.tight_layout()
        plt.show()

    # Accuracy Metrics
    def forecast_accuracy(self, forecast, actual):
        """Perform Forecast Accuracy Computation
        Parameters
        ----------
        forecast {Series}
            - forecast
        actual {Series}
            - actual

        Return
        ----------
        mape, mae, mse, rmse - metrics results

        """
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
        mae = np.mean(np.abs(forecast - actual))  # MAE
        mse = mean_squared_error(actual, forecast)  # MSE
        rmse = sqrt(mean_squared_error(actual, forecast))  # RMSE
        return ({'MAPE': mape, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})

    def naive_forecast(self, data):
        """Perform Forecast Accuracy Computation

        Parameters
        ----------
        data {Series}
            - time series data
        test_y {Series}
            - test

        Return
        ----------
        predictions {Series}
            - forecast
        test_y {Series}
            - test
        train_y {Series}
            - train
        """
        # Create lagged dataset
        data_val = DataFrame(data.values)
        results_df = concat([data_val.shift(1), data_val], axis=1)
        results_df.columns = ['t-1', 't+1']

        # split into train and test sets
        X = results_df.values
        train_size = int(len(X) * SPLIT)
        train, test = X[1:train_size], X[train_size:]
        train_X, train_y = train[:, 0], train[:, 1]
        test_X, test_y = test[:, 0], test[:, 1]

        # persistence model
        def model_persistence(x):
            return x

        # walk-forward validation
        predictions = list()
        for x in test_X:
            yhat = model_persistence(x)
            predictions.append(yhat)
        return (predictions, test_y, train_y)

    def SARIMA_forecast(self, train, test, order, seasonal_order):
        """Perform SARIMA Modeling

        Parameters
        ----------
        train {Series}
            - train data
        test {Series}
            - test data
        order {tuple}
            - selected order of (p,d,q)
        seasonal_order {tuple}
            - selected seasonal_order (P,D,Q)

        Return
        ----------
        forecast {Series}
            - forecasted data
        """
        print('\t SARIMA MODEL : In - Sample Forecasting \n')
        # Best model:  ARIMA(1,0,1)(0,1,1)[12]
        train_data = [x for x in train]
        forecast = []
        for t in range(len(test)):
            model = sm.tsa.statespace.SARIMAX(train_data,
                                              order=order,
                                              seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            result_fit = model_fit.forecast()
            y_hat = result_fit[0]
            forecast.append(float(y_hat))
            test_t = test[t]
            train_data.append(test_t)
        print('SARIMA Modeling DONE!')
        return (forecast)

    # Function for ARIMA Grid Search based on Brownlee, J. (2021, September 6).
    # Autoregression models for time series forecasting with python. Machine
    # Learning Mastery. Retrieved Sep 30, 2022, from
    # https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    def ARIMA_grid_search(self, data, p_values, d_values, q_values):
        """Perform ARIMA Grid Search

        Parameters
        ----------
        data {Series}
            - time series data
        p_values {range}
            - range of p values
        d_values {range}
            - range of d values
        q_values {range}
            - range of qvalues
        """

        # evaluate an ARIMA model for a given order (p,d,q)
        def evaluate_arima_model(X, arima_order):
            # prepare training dataset
            train_size = int(len(X) * SPLIT)
            train, test = X[0:train_size], X[train_size:]
            history = [x for x in train]
            # make predictions
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=arima_order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                history.append(test[t])
            # calculate out of sample error
            rmse = sqrt(mean_squared_error(test, predictions))
            return rmse

        # evaluate combinations of p, d and q values for an ARIMA model
        def evaluate_models(dataset, p_values, d_values, q_values):
            dataset = dataset.astype('float32')
            best_score, best_cfg = float("inf"), None
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        order = (p, d, q)
                        try:
                            rmse = evaluate_arima_model(dataset, order)
                            if rmse < best_score:
                                best_score, best_cfg = rmse, order
                            print('ARIMA%s RMSE=%.3f' % (order, rmse))
                        except:
                            continue
            print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

        evaluate_models(data.values, p_values, d_values, q_values)