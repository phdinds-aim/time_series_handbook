## Introduction
In the previous chapters we talked about Simplex Projection, a forecasting technique that looks for similar trends in the past to forecast the future by computing for nearest neighbors on an embedding. In this chapter, we discuss  Convergent Cross Mapping (CCM) also formulated by [Sugihara et al., 2012](https://science.sciencemag.org/content/338/6106/496) as a methodology that uses ideas from Simplex Projection to identify causality between variables in a complex dynamical system (e.g. ecosystem) using just time series data.

We will go through the key ideas of CCM, how it addresses the limitations of Granger causality, and the algorithm behind it. We will then test the CCM framework on simulated data where we will deliberately adjust the influence of one variable over the other. Finally, we will apply CCM on some real world data to infer the relationships between variables in a system.

### causal-ccm Package
`ccm_sugihara.ipynb` explains the CCM methodology in detail. If you wish to apply this in your own projects, install the framework using `pip install causal-ccm`. See `using_causal_ccm_package.ipynb` notebook for details how to use.