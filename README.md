# Time Series Handbook

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# How to use this reference
Insert instructions here

# Outline 
This handbook contains a variety of techniques that you can use for time series analysis -- from simple statistical models to some of the state-of-the-art algorithms as of writing. Here are the items that are covered in this material:
- Chapter 1: [Autoregressive integrated moving average](01_ARIMAandExponentialSmoothing)
- Chapter 2: [Linear Trend and Momentum Forecasting](02_LinearForecastingTrendandMomentumForecasting)
- Chapter 3: [Vector Autoregressive Methods](03_VectorAutoregressiveModels)
- Chapter 4: [Granger Causality](04_GrangerCausality)
- Chapter 5: [Simplex and S-map Projections](05_SimplexandSmapProjections)
- Chapter 6: [Convergent Cross Mapping and Sugihara Causality](06_ConvergentCrossMappingandSugiharaCausality)
- Chapter 7: [Frequency Analysis](07_FrequencyAnalysis)
- Chapter 8: [Winningest Methods](08_WinningestMethods)
    


# Getting Started: Setting up your virtual environment
To be able to run the contents of this repository, it is advised that you setup a virtual environment. You can install one via Anaconda or via Python's native `venv` module. 

##### Anaconda 
To set up a virtual environment called `atsa`, run the following in your terminal:

```bash
# this will create an anaconda environment
# called atsa in 'path/to/anaconda3/envs/'
conda create -n atsa
```

To activate and enter the environment, run `conda activate atsa`. To deactivate the environment, either run `conda deactivate atsa` or exit the terminal. 

```bash
# sanity check that the path to the python
# binary matches that of the anaconda env
# after you activate it
which python
# for example, on my machine, this prints
# $ '/Users/kevin/anaconda3/envs/sci/bin/python'
```

For more information on setting up your virtual evironment using Anaconda, please visit [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

#### Python venv
To set up a virtual environment called `atsa`, run the following in your terminal:

```bash
# this will create a virtual environment
# called cs231n in your home directory
python3 -m venv ~/atsa
```

To activate and enter the environment, run `source ~/atsa/bin/activate`. To deactivate the environment, either run `deactivate` or exit the terminal. Note that every time you want to work on the assignment, you should rerun `source ~/atsa/bin/activate`.

### Contributors
- Benjur Emmanuel Borja
- Gilbert Chua
- Francis Corpuz
- Vinni Dajac
- Sebastian Ibanez
- Prince Javier
- Marissa Pastor-Liponhay
- Maria Eloisa Ventura

