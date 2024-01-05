#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:07:34 2023

@author: s
"""

TrainingDataFile = "/Users/s/PhD/AllRuns_SingleTime_Train.csv"
TestDataFile = "/Users/s/PhD/AllRuns_SingleTime_Test.csv"
DataSplit = 'random' # split of data into training and test data 'manual' => Sub's separate files; 
                     #                                           'random' => Sub's separate files are first concatenated into one data set and then split randomly

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


#prepare data

# load data
def LoadData(csvFile):
    
    df = pd.read_csv(csvFile)

    df_Inputs = df.iloc[2:8,1:].astype(float)
    df_Inputs = df_Inputs.reset_index(drop=True)

    df_Targets = df.iloc[17:,1:].astype(float) # selects whole profile for each test sample
    df_Targets = df_Targets.reset_index(drop=True)
    
    y_alongshore = df.iloc[17:,0].astype(float).values

    return df_Inputs, df_Targets, y_alongshore
# this is the manula spec of training and test data by Sub's two files
X_train, y_train, y_alongshore = LoadData(TrainingDataFile)
X_test, y_test, y_alongshore = LoadData(TestDataFile)

#Smooth the data
def SmoothData(df,window_size=5,weights_type='gaussian', std_dev=1):
    
    '''
    df = dataframe
    window_size = size of moving average window
    weights_type = type of centred weight distribution
    std_dev = std dev of normal dist, increase for greater smooting

    '''
    
    if weights_type == 'gaussian':

        # Create centered weights based on the Gaussian distribution
        weights = norm.pdf(np.arange(window_size), loc=window_size // 2, scale=std_dev)
        df_Smoothed = df.rolling(window=window_size, center=True).apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)
    
    elif weights_type == 'equal': # equal weight moving average

        df_Smoothed = df.apply(lambda column: column.rolling(window=window_size, center=True).mean(), axis=0)

    # drop NaNs
    df_Smoothed = df_Smoothed.dropna()
    df_Smoothed = df_Smoothed.reset_index(drop=True)
    
    return df_Smoothed

#Do the smoothing, you can tune the degree of smoothing with std_dev and window size
std_dev = 1.5 # std dev of normal dist, increase for greater smoothing (I think .75 was Nick's default val)
window_size = 5 # smoothing window size 5=> center point and 2 either side
dropid = window_size // 2 # divides window_size by 2 and returns the quotient rounded down to the nearest whole number (integer).

y_train_smooth = SmoothData(y_train,window_size,'gaussian',std_dev=std_dev)
y_test_smooth = SmoothData(y_test,window_size,'gaussian',std_dev=std_dev)

#Plot raw vs smooth comparison
idx = 0 # test sample shoreline profile to plot raw vs smoothed

fig,axs=plt.subplots(1,2, figsize=(10, 5))
# training data
axs[0].plot(y_alongshore,y_train.iloc[:,idx],label='Raw Data')
axs[0].plot(y_alongshore[dropid:-dropid],y_train_smooth.iloc[:,idx], color='r', label='Smoothed Data')
axs[0].set_xlabel('Alongshore position of transect (m)')
axs[0].set_ylabel('Crosshore position of shoreline (m)')
axs[0].set_title('Training Data Sample')
axs[0].legend()

# testing data
axs[1].plot(y_alongshore,y_test.iloc[:,idx],label='Raw Data')
axs[1].plot(y_alongshore[dropid:-dropid],y_test_smooth.iloc[:,idx], color='r',label='Smoothed Data')
axs[1].set_xlabel('Alongshore position of transect (m)')
axs[1].set_ylabel('Crosshore position of shoreline (m)')
axs[1].set_title('Test Data Sample')
axs[1].legend()

plt.show()

#Drop edges from alongshore profile coordinates that are lost due to the moving widnow smoothing
y_alongshore = y_alongshore[dropid:-dropid] 

#Standardise data for ML input
#%% transpose dfs to get to work with model code below
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train_smooth.T, y_test_smooth.T

#%% Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaling parameters for scaling of single input test
scaling_mean = scaler.mean_
scaling_std = scaler.scale_

#Initialise dicts for skill metric collection
# Initialize dictionaries to store performance metrics
train_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
test_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

#Train the model
model_name = 'GaussianProcessRegressor'
model=GaussianProcessRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

#Get model predictions
# Predictions on training set
y_train_pred = model.predict(X_train_scaled)

# Predictions on testing set
y_test_pred = model.predict(X_test_scaled)

#Compute skill metrics
def getMetrics(actual,predicted):

    metrics = {}
    metrics['mae'] = mean_absolute_error(actual, predicted)
    metrics['mse'] = mean_squared_error(actual, predicted)
    metrics['rmse'] = np.sqrt(metrics['mse']) # root mean square error
    metrics['si'] = 100*metrics['rmse'] / (np.max(actual) - np.min(actual)) # scatter index
    metrics['r2'] = r2_score(actual, predicted)
    
    return metrics
# Calculate metrics for training set
train_metrics = getMetrics(y_train, y_train_pred)

# Calculate metrics for testing set
test_metrics = getMetrics(y_test, y_test_pred)

#Plot evaluation
# def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, mae, mse, r2, y0):
def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, metrics, x0, y0):
    
    ax.scatter(y_actual, y_pred, s=20, alpha=0.6, label=set_type)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to make axes square
    ax.set_xlim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    ax.set_ylim([min(y_actual.min().min(), y_pred.min().min()), max(y_actual.max().max(), y_pred.max().max())])
    formatted_string = set_type+'\n'+"\n".join([f"{key.upper()}: {value:.4f}" for key, value in metrics.items()])
    ax.text(x0, y0, formatted_string, transform=ax.transAxes)

    ax.grid(True)
    ax.legend()

#Skill scatter plot and shoreline profile comparison
test_index = 21 # index of test sample shoreline profile to compare

# skill scatter plot
plt.close('all')
fig, axes = plt.subplots(2, 1, figsize=(15, 9)) # the left plot is the model evaluation plot for all training and test data, the second will be the shoreline profile for a user specified test sample

# Plot actual vs. predicted for training set
plot_actual_vs_predicted(axes[0], y_train, y_train_pred, model_name, 'Training', train_metrics, 0.05, 0.5)

# Plot actual vs. predicted for testing set
plot_actual_vs_predicted(axes[0], y_test, y_test_pred, model_name, 'Testing', test_metrics, 0.65, 0.1)
axes[0].plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'k--', lw=2)  # Add 1:1 line

# select test sample to examine
X_input = [X_test.iloc[test_index]] 

# Scale the new input using the saved scaling parameters
X_input_scaled = (X_input - scaling_mean) / scaling_std
# get the actual test sample data
y_actual = y_test.iloc[test_index].values

# # run mode for the test sample
y_pred, std_prediction = model.predict(X_input_scaled, return_std=True)

# add the current test sample to the model scatter plot from above
axes[0].scatter(y_actual, y_pred, s=70, facecolors='none', color='r', label='Test Sample (see plot below)')
axes[0].legend()

# get text input paramerters for the plot title
input_par = X_test.iloc[test_index].values
input_dict = {'H' : input_par[0],
              'T' : input_par[1],
              'Dir' : input_par[2],
              'Dist' : input_par[3],
              'L' : input_par[4],
              'W' : input_par[5],
}
plot_title = 'Test Sample: '+', '.join([f"{key}={value}" for key, value in input_dict.items()])

# plot the shoreline profile
axes[1].plot(y_alongshore,y_actual, 'k', label='Actual')
axes[1].plot(y_alongshore,y_pred.T, 'r--', label='Predicted', lw=1)

# # add confidence intervals - not sure this is correct?
# ci_lower = y_pred - 1.96 * std_prediction  # 95% ci
# ci_upper = y_pred + 1.96 * std_prediction  # 95% ci
# axes[1].plot(y_alongshore,ci_upper.T,color='b', lw=0.5)
# axes[1].plot(y_alongshore,ci_lower.T,color='b', lw=0.5)
# # plot the cis
# axes[1].fill_between(
#     y_alongshore,
#     ci_lower.flatten(),
#     ci_upper.flatten(),
#     color='b',
#     alpha=0.2,
#     label=r"95% confidence interval",
# )

# appearance
axes[1].set_xlabel('Alongshore position of transect (m)')
axes[1].set_ylabel('Crosshore position of shoreline (m)')
axes[1].set_title(plot_title)
axes[1].grid(True)
axes[1].legend()
fig.tight_layout()

#General application of the model for prediction

#Define functions to check input data and run the model
def checkInputValidity(input_dict):
    '''

    Parameters
    ----------
    input_dict : dict of floats
        dict of the 6 input variables to the model H,T,Dir,Dist,L,W

    Returns
    -------
    data_ok : bool
        True is all parameters are in range
        False if one or more parameters are out of range
    out_of_bounds_keys : list
        list of variables that are out of range
        empty list of data_ok = True

    '''    
    # min allowable values of input parameters X_train.min()
    min_dict = {'H' : 0.5,
               'T' : 7.,
               'Dir' : 180., 
               'Dist' : 150.,
               'L' : 100.,
               'W' : 25.,
               }
    
    # max allowable values of input parameters X_train.ax()
    max_dict = {'H' : 2.5,
               'T' : 13.,
               'Dir' : 180., 
               'Dist' : 450.,
               'L' : 200.,
               'W' : 75.,
               }
    
    out_of_bounds_keys = []
    data_ok = True
    for key, value in input_dict.items():
        if key in min_dict and key in max_dict:
            if not (min_dict[key] <= value <= max_dict[key]):
                out_of_bounds_keys.append(key)
                data_ok = False
                
        else:
            # Handle case where key is not present in both min_values and max_values
            out_of_bounds_keys.append(key)
            data_ok = False
    
    return data_ok, out_of_bounds_keys
def ModelApplication(input_dict, model, scaling_mean, scaling_std, y_alongshore):

    # check if input data is within bounds of model parameter range
    data_ok, out_of_bounds_keys=checkInputValidity(input_dict)
    
    if data_ok:
        X_input = [list(input_dict.values())]    
        
        # Scale the provided input using the model's scaling parameters
        X_input_scaled = (X_input - scaling_mean) / scaling_std

        # run mode for the test sample
        y_pred = model.predict(X_input_scaled)
        # get the actual test sample data
        y_actual = y_test.iloc[test_index].values
    
        # add the current test sample to the model scatter plot from above
        fig, ax = plt.subplots(1, 1)#, figsize=(15, 9))
        plot_title = ', '.join([f"{key}={value}" for key, value in input_dict.items()])
    
        # plot the shoreline profile
        ax.plot(y_alongshore,y_pred.T,color='r', label='Predicted')
        ax.set_xlabel('Alongshore position of transect (m)')
        ax.set_ylabel('Crosshore position of shoreline (m)')
        ax.set_title(plot_title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        
    else:
        print('  *** ERROR *** \n The following input parameters are out of model range: '+','.join(out_of_bounds_keys) + '\n Refer to checkInputValidity(input_dict) function for min and max values for each parameter.')
    
    return

#Define model input parameters
# input dict
input_dict = {'H' : 1.0,
           'T' : 11.,
           'Dir' : 180., 
           'Dist' : 300.,
           'L' : 200.,
           'W' : 25.,
           }

#Run the model
ModelApplication(input_dict, model, scaling_mean, scaling_std, y_alongshore)

#Monte Carlo Sampling


#*** need to think more about usage and aim of this?
import numpy as np

def MonteCarloSampling(Nsamples, ParameterRanges, step_sizes=None):
    """
    Perform Monte Carlo sampling for each variable.

    Parameters:
    - Nsamples: Number of samples to generate.
    - ParameterRanges: Dictionary with parameter names as keys and [min, max] bounds as values.
    - step_sizes: Dictionary with parameter names as keys and desired step sizes as values.

    Returns:
    - parameter_samples: 2D array with sampled parameter values.
    """
    parameter_samples = np.zeros((Nsamples, len(ParameterRanges)))

    for i, (param, bounds) in enumerate(ParameterRanges.items()):
        if step_sizes and param in step_sizes:
            step_size = step_sizes[param]
            sampled_values = np.random.uniform(bounds[0], bounds[1], Nsamples)
            rounded_values = np.round(sampled_values / step_size) * step_size
            parameter_samples[:, i] = rounded_values
        else:
            parameter_samples[:, i] = np.random.uniform(bounds[0], bounds[1], Nsamples)

    return parameter_samples
Nsamples = 10

ParameterRanges = {'H' : [0.5,2.5],
                   'T' : [7., 13.],
                   'Dir' : [180., 180.],
                   'Dist' : [150., 450.],
                   'L' : [100., 200.],
                   'W' : [25., 75.],
                  }

# sampling intervals, the random samples are rounded to the nearest step size
step_sizes = {
    'H': 0.1,
    'T': 0.5,
    'Dir': 1.0,
    'Dist': 20,
    'L': 50,
    'W': 5,
}

# Perform Monte Carlo sampling with step sizes
ParameterSamples = MonteCarloSampling(Nsamples, ParameterRanges, step_sizes)

# display output - comment out to suppress
print('    H,    T,   Dir, Dist,  L,    W')
print('   ----------------------------------')
print(ParameterSamples)
#Run model using monte carlo samples
for ii in range(Nsamples):
    # define input_dict for current sample set
    input_dict = {'H' : np.round(ParameterSamples[ii,0],1),
                  'T' : np.round(ParameterSamples[ii,1],1),
                  'Dir' : np.round(ParameterSamples[ii,2],0), 
                  'Dist' : np.round(ParameterSamples[ii,3],0),
                  'L' : np.round(ParameterSamples[ii,4],0),
                  'W' : np.round(ParameterSamples[ii,5],0),
                  }
    
    # run the model
    ModelApplication(input_dict, model, scaling_mean, scaling_std, y_alongshore)
 