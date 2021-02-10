# Chapter 8: Winningest Methods in Time Series Forecasting
In previous sections, we examined several models used in time series forecasting such as ARIMA, VAR, and Exponential Smoothing methods. While the main advantage of traditional statistical methods is their ability to perform more sophisticated inference tasks directly (e.g. hypothesis testing on parameters, causality testing), they usually lack predictive power because of their rigid assumptions. That is not to say that they are necessarily inferior when it comes to forecasting, but rather they are typically used as performance benchmarks.

In this section, we demonstrate several of the fundamental ideas and approaches used in the recently concluded M5 Competition where challengers from all over the world competed in building time series forecasting models for both accuracy and uncertainty prediction tasks. Specifically, we explore the machine learning model that majority of the competition's winners utilized: LightGBM, a tree-based gradient boosting framework designed for speed and efficiency.

### How to use these notebooks
Please access the notebooks in this sequence:

- lightgbm_m5_forecasting.ipynb

- lightgbm_m5_tuning.ipynb

- lightgbm_jena_forecasting.ipynb

### M5 Dataset
The M5 dataset consists of Walmart sales data. Specifically, daily unit sales of 3,049 products, classified in 3 product categories (Hobbies, Foods, and Household), and 7 product departments in which the previously mentioned categories are disaggregated. The products are sold across 10 stores, located in 3 States (California, Texas, and Wisconsin).

To download the dataset, you may do so at the following link: https://www.kaggle.com/c/m5-forecasting-accuracy

### Chapter Outline

1. M5 Dataset

2. Pre-processing

3. One-Step Prediction

4. Multi-Step Prediction

5. Feature Importance

6. Summary
