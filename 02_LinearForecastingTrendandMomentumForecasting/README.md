# Chapter 2: Linear, Trend, and Momentum Forecasting

In this chapter we introduce basic tools on forecasting, which utilize simple algebraic formula. In the previous chapter, ARIMA was discussed where the future values of a time series are forecasted using its past or lagged values. It was shown that ARIMA can only be applied after removing the trend and seasonality of the data. We note however that for some forecasting tools, the trend is relevant and is part of the formula for prediction. In this work, forecasting will be demonstrated while making use of the relationships and trends in the data.

In the first half of this notebook, we demonstrate forecasting by fitting time series data with linear regression. For the second half, we demonstrate that by using the trends of the time series data such as moving averages, we can predict the possible future direction of the trend using momentum forecasting.

Lastly, it is important to note that the concept of moving average (MA) in ARIMA is not the same in this chapter since the moving average that will be discussed is just the classical definition of MA.