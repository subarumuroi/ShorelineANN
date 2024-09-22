#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:25:49 2023

@author: s
"""


allFile = "/Users/s/PhD/AllRuns.csv"


Adam_learning_rate = 0.0066
Nepochs = 5000
Nneurons_Dense = 1200
activation = 'relu'
batch = 32

verbose = 1 # 1 => display model training/prediction progress data to console or 0 => don't


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import norm


from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
x_train, y_train, y_alongshore = LoadData(allFile)


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

#Drop edges from alongshore profile coordinates that are lost due to the moving widnow smoothing
y_alongshore = y_alongshore[dropid:-dropid] 


#Standardise data for ML input
#%% transpose dfs to get to work with model code below
x_train, y_train, = x_train.T, y_train_smooth.T

#%% Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)


# Save scaling parameters for scaling of single input test
scaling_mean = scaler.mean_
scaling_std = scaler.scale_

#Initialise dicts for skill metric collection
# Initialize dictionaries to store performance metrics
train_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
test_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}






##took neural net from other here



# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the validation loss
                               patience=1200,           # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity



#Compute skill metrics
def getMetrics(actual,predicted):

    metrics = {}
    metrics['mae'] = mean_absolute_error(actual, predicted)
    metrics['mse'] = mean_squared_error(actual, predicted)
    metrics['rmse'] = np.sqrt(metrics['mse']) # root mean square error
    metrics['si'] = 100*metrics['rmse'] / (np.max(actual) - np.min(actual)) # scatter index
    metrics['r2'] = r2_score(actual, predicted)
    
    return metrics

model_name = 'Neural Net'

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    #tf.keras.layers.Dense(Nneurons_Dense, activation='tanh'),
    tf.keras.layers.Dense(51)  #output layer with 51var
])
  
  #optimizer
opt = tf.keras.optimizers.legacy.Adam(learning_rate = Adam_learning_rate, clipnorm = 1)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

# define k-fold measures
k = 5
kf = KFold(n_splits=k, shuffle=True)

# Initialize lists to store performance metrics for each fold
train_mae_list, train_mse_list, train_r2_list = [], [], []
test_mae_list, test_mse_list, test_r2_list = [], [], []

   # Iterate over the folds
for train_index, test_index in kf.split(x_train_scaled, y_train):
       # Split the data into training and validation sets for this fold
       x_cv_train, x_cv_val = x_train_scaled[train_index], x_train_scaled[test_index]

# Iterate over the folds
for train_index, test_index in kf.split(y_train):
    # Split the data into training and validation sets for this fold
    y_cv_train, y_cv_val = y_train[train_index], y_train[test_index]

    
# Iterate over the folds
for train_index, test_index in kf.split(x_train_scaled, y_train):
    # Split the data into training and validation sets for this fold
    x_cv_train, x_cv_val = x_train_scaled[train_index], x_train_scaled[test_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[test_index]
    
    # Train the model on the training data for this fold
    fit = model.fit(x_cv_train, y_cv_train, epochs=Nepochs, batch_size=batch, verbose=verbose, validation_data=(x_cv_val, y_cv_val), callbacks=[early_stopping])
    
    # Get model predictions on training and validation sets
    y_cv_train_pred = model.predict(x_cv_train)
    y_cv_val_pred = model.predict(x_cv_val)
    
    # Calculate performance metrics for training set
    train_metrics = getMetrics(y_cv_train, y_cv_train_pred)
    train_mae_list.append(train_metrics['mae'])
    train_mse_list.append(train_metrics['mse'])
    train_r2_list.append(train_metrics['r2'])
    
    # Calculate performance metrics for validation set
    val_metrics = getMetrics(y_cv_val, y_cv_val_pred)
    test_mae_list.append(val_metrics['mae'])
    test_mse_list.append(val_metrics['mse'])
    test_r2_list.append(val_metrics['r2'])

# Calculate mean performance metrics across all folds
mean_train_mae = np.mean(train_mae_list)
mean_train_mse = np.mean(train_mse_list)
mean_train_r2 = np.mean(train_r2_list)

mean_test_mae = np.mean(test_mae_list)
mean_test_mse = np.mean(test_mse_list)
mean_test_r2 = np.mean(test_r2_list)

# Print mean performance metrics
print("Mean Training MAE:", mean_train_mae)
print("Mean Training MSE:", mean_train_mse)
print("Mean Training R2:", mean_train_r2)

print("Mean Test MAE:", mean_test_mae)
print("Mean Test MSE:", mean_test_mse)
print("Mean Test R2:", mean_test_r2)


'''
#Train the model
model_name = 'GaussianProcessRegressor'
model=GaussianProcessRegressor(random_state=42)
fit = model.fit(x_train_scaled, y_train)

'''

'''

#Get model predictions
# Predictions on training set
y_train_pred = model.predict(x_train_scaled)

# Predictions on testing set
y_test_pred = model.predict(x_test_scaled)

'''



'''
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
'''

'''
# Calculate metrics for training set
train_metrics = getMetrics(y_train, y_train_pred)

# Calculate metrics for testing set
test_metrics = getMetrics(y_test, y_test_pred)
'''

'''
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

'''

