#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:25:49 2023

@author: s
"""

trainFile = "/Users/s/PhD/AllRuns_SingleTime_Train.csv"
testFile = "/Users/s/PhD/AllRuns_SingleTime_Test.csv"


Adam_learning_rate = 0.0066
Nepochs = 5000
Nneurons_Dense = 1200
activation = 'relu'
batch = 32

verbose = 1 # 1 => display model training/prediction progress data to console or 0 => don't

DataSplit = 'random' # split of data into training and test data 'manual' => Sub's separate files; 
                     #                                           'random' => Sub's separate files are first concatenated into one data set and then split randomly

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold


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
'''
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
'''
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



######################3
##Pick one of the 3 models below to execute: Neural network, Gaussian Process Regressor, or Decision Tree
##Default set to NN but other models also work 

##took neural net from other here

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
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



model_name = 'Deep Neural Net 2-layer'


# convert the history to a dataframe for plotting  (ONLY FOR NN)
history_dropout_df = pd.DataFrame.from_dict(fit.history)


# Plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Neural Network Model')

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
#Train the model
model_name = 'GaussianProcessRegressor'
model=GaussianProcessRegressor(random_state=42)
fit = model.fit(x_train_scaled, y_train)

'''

'''
from sklearn.tree import DecisionTreeRegressor

model_name = 'DecisionTreeRegressor'
model = DecisionTreeRegressor(random_state=42)
fit = model.fit(x_train_scaled, y_train)
'''
'''
from sklearn.ensemble import RandomForestRegressor

model_name = 'RandomForestRegressor'
# Initialize the Random Forest Regressor with desired parameters
model = RandomForestRegressor(n_estimators=1000, random_state=42)
fit = model.fit(x_train_scaled, y_train)
'''
'''
from sklearn.neighbors import KNeighborsRegressor

model_name = 'KNeighborsRegressor'
# Initialize the KNN Regressor with desired parameters
model = KNeighborsRegressor(n_neighbors=1)  # You can adjust the number of neighbors
fit = model.fit(x_train_scaled, y_train)
'''


##############################################


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
Nsamples = 1

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
    
'''
import math 
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50], # Random choice, can be adjusted to experiment
}


from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Function to create Keras model
def create_model(learning_rate=0.01, batch_size=32):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
        tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
        tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
        tf.keras.layers.Dense(51)
    ])
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model

# Create KerasRegressor
keras_wrapper = KerasRegressor(build_fn=create_model, verbose=0)

# Define parameters for grid search

param_grid = {
    'learning_rate': [0.005,0.0075, 0.01],
    'batch_size': [16, 32, 64],
    'epochs': [1000, 2000, 3000]  # You can adjust the number of epochs as needed
}

# Perform grid search
grid_search = GridSearchCV(estimator=keras_wrapper, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(x_train_scaled, y_train)

# Get best parameters
best_learning_rate = grid_result.best_params_['learning_rate']
best_batch_size = grid_result.best_params_['batch_size']
best_epochs = grid_result.best_params_['epochs']

# Train the best model
best_model = create_model(learning_rate=best_learning_rate, batch_size=best_batch_size)
best_model.fit(x_train_scaled, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=verbose, validation_data=(x_test_scaled, y_test), callbacks=[early_stopping])

# Evaluate the best model
test_loss, test_accuracy = best_model.evaluate(x_test_scaled, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    return lr * math.exp(-0.1)  # You can adjust the decay rate to test different convergence rates

# Create a LearningRateScheduler callback
# Note: A 'callback' essentially allows specification of certain model 'behaviour' at particular points in training
lr_callback = LearningRateScheduler(lr_scheduler)

# Early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=10,  # This is the of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best model weights
)

# Train model with the optimal learning rate, batch size, learning rate scheduler, and early stopping
history = best_model.fit(
    x_train_scaled, y_train,
    epochs=50,  # Adjust the number of epochs as needed
    batch_size=best_batch_size,
    validation_data=(x_test_scaled, y_test),  # Use the test set for validation
    callbacks=[lr_callback, early_stopping_callback]
)

# Plot the learning rate range test results (loss vs. learning rate)
learning_rates = [10 ** i for i in range(-5, 0)]  # A likely range of learning rates (would not expect 10e-6 or lower)
losses = []

for lr in learning_rates:
    model = model(learning_rate=lr, batch_size=best_batch_size)
    history = model.fit(
        x_train_scaled, y_train,
        epochs=2000,  # I have used a small epoch size. Your speedy Mac could probably handle more :)
        batch_size=best_batch_size,
        validation_data=(x_test_scaled, y_test),
        verbose=1
    )
    losses.append(history.history['val_loss'][-1])

plt.semilogx(learning_rates, losses, marker='o')
plt.title('Learning Rate Range Test')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.show()

# Final eval, printing best params
y_pred = best_model.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Batch Size: {best_batch_size}')
print(f'Mean Squared Error on Test Data: {mse}')
'''
