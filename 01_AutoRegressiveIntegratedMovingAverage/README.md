# Chapter 1: AutoRegressive Integrated Moving Average (ARIMA)

In this notebook, we will introduce our first approach to time-series forecasting which is **ARIMA** or AutoRegressive Integrated Moving Average. ARIMA, or AutoRegressive Integrated Moving Average, is a set of models that explains a time series using its own previous values given by the lags (**A**uto**R**egressive) and lagged errors (**M**oving **A**verage) while considering stationarity corrected by differencing (oppossite of **I**ntegration.) In other words, ARIMA assumes that the time series is described by autocorrelations in the data rather than trends and seasonality.

This notebook will discuss:

1. Definition and Formulation of ARIMA models

2. Model Parameters (p, d, and q) and Special Cases of ARIMA models

3. Model Statistics and How to Interpret

4. Implementation and Forecasting using ARIMA

#### Datasets used:
- *Synthetic Data* (Filename: [`../data/wwwusage.csv`](https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv)) 

- *Climate Data* (Filename: `../data/jena_climate_2009_2016.csv"`)

- *Household Electric Power Consumption* (Filename: `../data/power_consumption/household_power_consumption.txt"`) (Learning, U. C. I. M. (2016, August 23). Household Electric Power Consumption. Kaggle. Retrieved November 2022, from https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set?select=household_power_consumption.txt)
