# Chapter 5: Empirical Dynamic Modeling (Simplex and SMap Projections)

In the previous sections, we looked at the different methods to characterize a time-series and other statistical operations that we can execute to perform predictions. Many of these methods involve calculating for the models that would best fit the time series, extracting the optimal parameters that would describe the data with the least error possible. However, many real world processes exhibit nonlinear, complex, dynamic characteristics, necessitating the need of other methods that can accommodate as such.

In this section, we will introduce and discuss methods that uses empirical models instead of complex, parametized, and hypothesized equations. Using raw time series data, we will try to reconstruct the underlying mechanisms that might be too complex, noisy, or dynamic to be captured by equations. This method proposes a altenatively more flexible approach in working and predicting with dynamic systems.


## This Notebook will discuss the following:
- Introduction to Empirical Dynamic Modelling
- Visualization of EDM Prediction with Chaotic Time Series
- Lenz' Attractor
- Taken's Theorem / State-Space Reconstruction (SSR)
- Simplex Projection
- Determination of Optimal Embedding Values
- Differentiation Noisy Signals from Chaotic Signals
- S-Map Projection (Sequentially Locally Weighted Global Linear Map)
