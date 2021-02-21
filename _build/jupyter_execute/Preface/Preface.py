# Preface: Introduction to Time Series Analysis

This handbook extensively covers time series analysis and forecasting, delving from the most fundamental methods to the state-of-the-art. The handbook was made in Python and is designed such that readers can both learn the theory and apply them to real-world problems. Although chapters were made to be stand alone, it is recommended that readers go through the first few chapters to be able to fully appreciate the latter chapters. Moreover, the 
__[Jena climate dataset](https://www.kaggle.com/stytch16/jena-climate-2009-2016)__ is used across several chapters, with a summary of the performance of the models used at the end.

The handbook is structured as follows: in the first part, classical forecasting methods are discussed in detail. The middle part is then dedicated to dynamical forecasting methods and as well as causality and correlations, topics that are particularly essential in understanding the intricacies of time series forecasting. Finally, the last part shows a glimpse into the current trends and open problems in time series forecasting and modeling.

The aim of this handbook is to serve as a practitioner’s guide to forecasting, enabling them to better understand relationships in signals. It is made for an audience with a solid background in Statistics and Mathematics, as well as a basic knowledge of Python. Familiarity with Machine Learning methods is a plus, especially for the later chapters. 





```{toctree}
:hidden:
:titlesonly:


../00_Introduction/00_Introduction
../01_AutoRegressiveIntegratedMovingAverage/01_AutoRegressiveIntegratedMovingAverage
../02_LinearForecastingTrendandMomentumForecasting/02_LinearTrendandMomentumForecasting
../03_VectorAutoregressiveModels/03_VectorAutoregressiveMethods
../04_GrangerCausality/04_GrangerCausality
../05_SimplexandSmapProjections/05_Empirical Dynamic Modelling (Simplex and SMap_Projections)
../06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara
../07_CrosscorrelationsFourierTransformandWaveletTransform/07_CrosscorrelationsFourierTransformandWaveletTransform
../08_WinningestMethods/lightgbm_m5_forecasting
../data/README
```
