# Chapter 7: Cross-Correlations, Fourier Transform, and Wavelet Transform

Gaining a deeper undersanding of causality, we look at time series forecasting through another lens.

In this chapter, we will take a different approach to how we analzye time series that is complementary to forecasting. Previously, methods of explaining such as the Granger Causality, S-mapping, and Cross-mapping, focused on the time domain - the values of the time series when measured over time or in its phase space. While both are useful for many tasks, it can be often useful to transform these time domain measurements to unearth patterns which are difficult to tease out. Specifically, we want to look at the frequency domain to both analyze the dynamics and perform pre-processing techniques that may be used to modify real-world datasets.

We will be analyzing the dynamics of time series not exactly to make forecasts, but to understand them in terms of their frequencies in complement to the previous methods of causality and explainability presented.

We introduce three techniques:

1. Cross-correlations

2. Fourier Transform

3. Wavelet Transform

and test their use on the Jena Climate Dataset (2009-2016) along with a handful of other datasets.
