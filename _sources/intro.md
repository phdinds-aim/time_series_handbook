# Time Series Handbook

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Outline 
This handbook contains a variety of techniques that you can use for time series analysis -- from simple statistical models to some of the state-of-the-art algorithms as of writing. Here are the items that are covered in this material:
- Preface: [Advanced Time Series Analysis](00_Preface)
- Chapter 1: [Autoregressive integrated moving average](01_AutoRegressiveIntegratedMovingAverage)
- Chapter 2: [Linear Trend and Momentum Forecasting](02_LinearForecastingTrendandMomentumForecasting)
- Chapter 3: [Vector Autoregressive Methods](03_VectorAutoregressiveModels)
- Chapter 4: [Granger Causality](04_GrangerCausality)
- Chapter 5: [Simplex and S-map Projections](05_SimplexandSmapProjections)
- Chapter 6: [Convergent Cross Mapping and Sugihara Causality](06_ConvergentCrossMappingandSugiharaCausality)
- Chapter 7: [Cross-Correlations, Fourier Transform and Wavelet Transform](07_CrosscorrelationsFourierTransformandWaveletTransform)
- Chapter 8: [Winningest Methods](08_WinningestMethods)
    

# How to use this reference
Each of the chapters mentioned above includes Jupyter notebook/s that contain/s the discussion of each topic (background, limitations, applications). Most of the datasets used in the handbook are included in this repository, and the details of each are described in the [data folder](data). 

## Setting up your virtual environment
To be able to run the contents of this repository, it is advised that you setup a virtual environment. You can install one via Anaconda or via Python's native `venv` module. 

##### Anaconda 
To set up a virtual environment called `atsa`, run the following in your terminal:

```bash
# this will create an anaconda environment
# called atsa in 'path/to/anaconda3/envs/'
conda create -n atsa
```

To activate and enter the environment, run `conda activate atsa`. To deactivate the environment, either run `conda deactivate atsa` or exit the terminal. For more information on setting up your virtual evironment using Anaconda, please visit [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

##### Python venv
To set up a virtual environment called `atsa`, run the following in your terminal:

```bash
# this will create a virtual environment
# called atsa in your home directory
python3 -m venv ~/atsa
```

To activate and enter the environment, run `source ~/atsa/bin/activate`. To deactivate the environment, either run `deactivate` or exit the terminal. Note that every time you want to work on the assignment, you should rerun `source ~/atsa/bin/activate`.

## Rendering the notebooks
To view the notebooks outside of Github without setting up a repository or installing any software, you may use [The Jupyter Notebook Viewer](https://nbviewer.jupyter.org/).  
- Open `https://nbviewer.jupyter.org/`
- Paste the link to the notebook. 

When a notebook rendered in nbviewer appears differently from the one rendered github, just append `?flush_cache=true` to the end of the nbviewer URL to force it to rerender.

# Contributors
- Benjur Emmanuel Borja
- Gilbert Chua
- Francis James Corpuz
- Carlo Vincienzo Dajac
- Sebastian Ibanez
- Prince Javier
- Marissa Pastor-Liponhay
- Maria Eloisa Ventura

