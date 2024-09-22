#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:49:04 2024

@author: s
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:25:49 2023

@author: s
"""

trainFile = "/Users/s/PhD/AllRuns_SingleTime_Train.csv"
testFile = "/Users/s/PhD/AllRuns_SingleTime_Test.csv"


Adam_learning_rate = 0.0066
Nepochs = 2000
Nneurons_Dense = 320
activation = 'relu'
batch = 32

verbose = 0 # 1 => display model training/prediction progress data to console or 0 => don't


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Set random seed
np.random.seed(42)
tf.random.set_seed(42)


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
x_train, y_train, y_alongshore = LoadData(trainFile)
x_test, y_test, y_alongshore = LoadData(testFile)

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
std_dev = .75 # std dev of normal dist, increase for greater smoothing (I think .75 was Nick's default val)
window_size = 5 # smoothing window size 5=> center point and 2 either side
dropid = window_size // 2 # divides window_size by 2 and returns the quotient rounded down to the nearest whole number (integer).

y_train_smooth = SmoothData(y_train,window_size,'gaussian',std_dev=std_dev)
y_test_smooth = SmoothData(y_test,window_size,'gaussian',std_dev=std_dev)

#Plot raw vs smooth comparison
idx = 10 # test sample shoreline profile to plot raw vs smoothed

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
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train_smooth.T, y_test_smooth.T

#%% Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Save scaling parameters for scaling of single input test
scaling_mean = scaler.mean_
scaling_std = scaler.scale_

#Initialise dicts for skill metric collection
# Initialize dictionaries to store performance metrics
train_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
test_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}




'''

##NEURAL NETWORK DESIGN VARIABLES AT TOP

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    #tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    #tf.keras.layers.Dense(Nneurons_Dense, activation='tanh'),
    tf.keras.layers.Dense(51)  #output layer with 1var
])
#optimizer
opt = tf.keras.optimizers.legacy.Adam(learning_rate = Adam_learning_rate, clipnorm = 1)
#compile the model
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])


# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the validation loss
                               patience=800,           # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity


fit=model.fit(x_train_scaled, y_train, epochs=Nepochs, batch_size=batch, verbose=verbose, validation_data=(x_test_scaled, y_test), callbacks = [early_stopping])


model_name = 'Neural Network Model'


# convert the history to a dataframe for plotting 
history_dropout_df = pd.DataFrame.from_dict(fit.history)


# Plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('DNN Model')

# Plot Loss
axes[0].set_ylim(0, 10)
sns.lineplot(ax=axes[0], data=history_dropout_df[['loss', 'val_loss']])
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

# Plot Accuracy
sns.lineplot(ax=axes[1], data=history_dropout_df[['accuracy', 'val_accuracy']])
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')

# Loss plot
plt.figure()
plt.plot(history_dropout_df['loss'], label='Training Loss')
plt.plot(history_dropout_df['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the combined plots
plt.show()


#nerual net ends here
'''
###load best nn model
'''
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model_NN_best.h5')
'''



#Train the GPR model
model_name = 'GaussianProcessRegressor'
model=GaussianProcessRegressor(alpha = 1e-7, kernel = Matern(nu=0.8), random_state=42)
fit = model.fit(x_train_scaled, y_train)
#end GPR model




#Get model predictions
# Predictions on training set
y_train_pred = model.predict(x_train_scaled)

# Predictions on testing set
y_test_pred = model.predict(x_test_scaled)








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
    ax.set_xlabel('Actual Shoreline Change (m)')
    ax.set_ylabel('Predicted Shoreline Change (m)')
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
x_input = [x_test.iloc[test_index]] 

# Scale the new input using the saved scaling parameters
x_input_scaled = (x_input - scaling_mean) / scaling_std
# get the actual test sample data
y_actual = y_test.iloc[test_index].values

# # run mode for the test sample
y_pred = model.predict(x_input_scaled)

# add the current test sample to the model scatter plot from above
axes[0].scatter(y_actual, y_pred, s=70, facecolors='none', color='r', label='Test Sample (see plot below)')
axes[0].legend()

# get text input paramerters for the plot title
input_par = x_test.iloc[test_index].values
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



def ModelApplication(input_dict, model, scaling_mean, scaling_std, y_alongshore, avg_value, avg_params, max_value, max_params, min_value, min_params):
    # check if input data is within bounds of model parameter range
    data_ok, _ = checkInputValidity(input_dict)
    
    if data_ok:
        X_input = [list(input_dict.values())]    
        
        # Scale the provided input using the model's scaling parameters
        X_input_scaled = (X_input - scaling_mean) / scaling_std

        # Run model for the test sample
        y_pred = model.predict(X_input_scaled)

        # Average the predicted values
        #avg_pred = np.mean(y_pred)
        
       
        avg_pred =  np.sum(np.abs(y_pred))


        # Check if the current prediction is the highest
        if avg_pred < avg_value:
            avg_value = avg_pred
            avg_params = input_dict.copy()
        
        
        
        # Find the maximum value within the series of 51 values
        max_series_value = np.max(y_pred)
        
        #check if current prediction is the largest (causes most accretion)
        if max_series_value > max_value:
            max_value = max_series_value
            max_params = input_dict.copy()
            
            
            
        # Find the minimum value within the seris of 51 values    
        min_series_value = np.min(y_pred)
        # Check if the 26th value of the series is positive
        if y_pred[0][25] > 0:  # Assuming y_pred is a 2D array with shape (1, 51)
            # Check if the current prediction is the largest (causes the least erosion)
            if min_series_value > min_value:
                min_value = min_series_value
                min_params = input_dict.copy()
       
    
    return avg_value, avg_params, max_value, max_params, min_value, min_params




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



Nsamples = 10000

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
    'T': 0.1,
    'Dir': 1.0,
    'Dist': 5,
    'L': 5,
    'W': 5,
}

# Perform Monte Carlo sampling with step sizes
ParameterSamples = MonteCarloSampling(Nsamples, ParameterRanges, step_sizes)

# Define the values of 'H' and 'T' that you want to hold constant
constant_H = 1.5
constant_T = 10


# Initialize variables to keep track of the highest and lowest values and their corresponding parameters
avg_value = float('inf')
avg_params = {}
max_value = -float('inf')
max_params = {}
min_value = -float('inf')
min_params = {}

# Iterate through samples
for ii in range(Nsamples):
    # Define input_dict for current sample set
    input_dict = {'H' : constant_H, #np.round(ParameterSamples[ii,0],1),
                  'T' : constant_T, #np.round(ParameterSamples[ii,1],1),
                  'Dir' : np.round(ParameterSamples[ii,2],0), 
                  'Dist' : np.round(ParameterSamples[ii,3],0),
                  'L' : np.round(ParameterSamples[ii,4],0),
                  'W' : np.round(ParameterSamples[ii,5],0),
                 }



    avg_value, avg_params, max_value, max_params, min_value, min_params = ModelApplication(input_dict, model, scaling_mean, scaling_std, y_alongshore, avg_value, avg_params, max_value, max_params, min_value, min_params)

# Print the samples with highest average sediment change
print("Sample with the minimum absolute change")
print(avg_params)
print("Predicted value:", avg_value)
    
# Print the sample with the highest predicted value
print("\nSample with the highest predicted value:")
print(max_params)
print("Predicted value:", max_value)

# Print the sample with the lowest predicted value within the series of 51 values
print("\nSample with the lowest predicted value within the series of 51 values:")
print(min_params)
print("Predicted value:", min_value)





