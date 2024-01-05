#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:15:10 2023

@author: s
"""

#User Inputs
TrainingDataFile = "/Users/s/PhD/AllRuns_SingleTime_Train.csv"
TestDataFile = "/Users/s/PhD/AllRuns_SingleTime_Test.csv"
DataSplit = 'random' # split of data into training and test data 'manual' => Sub's separate files; 
                     #                                           'random' => Sub's separate files are first concatenated into one data set and then split randomly
#Import required packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#from sklearn_SUB import *

import matplotlib.pyplot as plt
plt.close('all')


#Prepare Data

#Load data
X_train, y_train = LoadData(TrainingDataFile)
X_test, y_test = LoadData(TestDataFile)
if DataSplit == 'random':
    X_all = pd.concat([X_train, X_test],axis=0)
    X_all = X_all.reset_index(drop=True)
    y_all = pd.concat([y_train, y_test],axis=0)
    y_all = y_all.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=99)
    
#Standardise features

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Develop and run ML Models
#Function to loop over models
def CompileRunAndEvaluateModels(models, X_train_scaled, y_train, y_test):
    
    train_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}
    test_metrics = {'Model': [], 'MAE': [], 'MSE': [], 'R2': []}

    num_models = len(models)
    fig_count = 0

    # Loop through each model
    for i, (model_name, model) in enumerate(models.items()):
        # print(model_name)
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predictions on training set
        y_train_pred = model.predict(X_train_scaled)

        # Predictions on testing set
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics for training set
        mae_train, mse_train, r2_train = getMetrics(y_train, y_train_pred)

        # Calculate metrics for testing set
        mae_test, mse_test, r2_test = getMetrics(y_test, y_test_pred)

        # Store metrics in dictionaries
        train_metrics['Model'].append(model_name)
        train_metrics['MAE'].append(mae_train)
        train_metrics['MSE'].append(mse_train)
        train_metrics['R2'].append(r2_train)

        test_metrics['Model'].append(model_name)
        test_metrics['MAE'].append(mae_test)
        test_metrics['MSE'].append(mse_test)
        test_metrics['R2'].append(r2_test)

        if i % 4 == 0:
            # Create a new 2x2 subplot figure for each set of training and testing plots
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig_count += 1

        # Plot actual vs. predicted for training set
        plot_actual_vs_predicted(axes[i % 4 // 2, i % 2], y_train, y_train_pred, model_name, 'Training', mae_train, mse_train, r2_train, 0.7)

        # Plot actual vs. predicted for testing set
        plot_actual_vs_predicted(axes[i % 4 // 2, i % 2], y_test, y_test_pred, model_name, 'Testing', mae_test, mse_test, r2_test, 0.5)

        if (i + 1) % 4 == 0 or i == num_models - 1:
            # Adjust layout for the last figure in each set
            fig.tight_layout()
            
    # Convert metrics dictionaries to DataFrames
    train_metrics_df = pd.DataFrame(train_metrics)
    train_metrics_df = train_metrics_df.set_index('Model')
    train_metrics_df = addMetricRank(train_metrics_df)

    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df = test_metrics_df.set_index('Model')
    test_metrics_df = addMetricRank(test_metrics_df)
            
    return train_metrics_df, test_metrics_df
#Run comparison of models
# Import models with default values
models = {
    
    # gaussian
    'Gaussian Process': GaussianProcessRegressor(),
    
    # neural network
    'MLPRegressor' : MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000),

    # tree models
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'HistGradientBoostingRegressor' : HistGradientBoostingRegressor(),
    
    'ExtraTreesRegressor' : ExtraTreesRegressor(),
    'AdaBoostRegressor' : AdaBoostRegressor(),

    # the models below are no good!

    # linear models
    'Linear Regression': LinearRegression(),
    'Ridge' : Ridge(),
    'BayesianRidge' : BayesianRidge(),
    'PassiveAggressiveRegressor' : PassiveAggressiveRegressor(),

    # Support Vector Machines
    'SVR': SVR(),
    'NuSVR' : NuSVR(),
    'LinearSVR' : LinearSVR(),
        
    # neighbours
    'KNeighborsRegressor': KNeighborsRegressor(),
    # 'RadiusNeighborsRegressor' : RadiusNeighborsRegressor(),
    
}

train_metrics_df, test_metrics_df = CompileRunAndEvaluateModels(models, X_train_scaled, y_train, y_test)

# Display training metrics
print("")
print("------------------------")
print("Model Skill and Rankings")
print("------------------------")
print("")
print("Training Metrics:")
print("")
print(train_metrics_df)
# train_metrics_df.to_csv('metrics_train.csv')

# Display testing metrics
print("\nTesting Metrics:")
print("")
print(test_metrics_df)
print("")
