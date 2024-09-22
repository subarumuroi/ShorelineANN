#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:31:40 2024

@author: s

This code is used to test pretrained Gaussian process regressor and neural
network models and see a graph of its prediction on a the input parameters 
given in 'x_input'. The bounds that can be tested in x_input is defined by 
the code that trains the models and is contained in the Monte Carlo Analysis 
section
"""
import joblib 
from tensorflow.keras.models import load_model

model_nn= load_model('model_NN_best.h5')
model_gpr= joblib.load('gpr_model_used_for_MC.pkl')

# Scale the new input using the saved scaling parameters
x_input = [[1.5, 10, 180, 230, 100, 25]]
x_input_scaled = (x_input - scaling_mean) / scaling_std

# Predict shoreline output using the neural network model
y_pred_nn = model_nn.predict(x_input_scaled)

# Predict shoreline output using the Gaussian Process Regressor model
y_pred_gpr = model_gpr.predict(x_input_scaled)

# Plot the shoreline profile comparison
plt.close('all')
fig, axes = plt.subplots(figsize=(15, 6))

# Get text input parameters for the plot title
input_par = x_input[0]
input_dict = {'H': input_par[0], 'T': input_par[1], 'Dir': input_par[2], 'Dist': input_par[3], 'L': input_par[4], 'W': input_par[5]}
plot_title = 'Test Sample: ' + ', '.join([f"{key}={value}" for key, value in input_dict.items()])

# Plot the shoreline profile
axes.plot(y_alongshore, y_pred_nn[0], 'r--', label='Predicted nn', lw=1)  # nn
axes.plot(y_alongshore, y_pred_gpr[0], 'g--', label='Predicted gpr', lw=1)  # gpr
axes.set_xlabel('Alongshore position of transect (m)')
axes.set_ylabel('Crosshore position of shoreline (m)')
axes.set_title(plot_title)
axes.grid(True)
axes.legend()
fig.tight_layout()
plt.show()