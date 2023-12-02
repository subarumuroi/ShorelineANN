# Sensitivity analysis script
# --------------------
# Ensure primary code has run and a .h5 extension model is available in the same directory as this script on execution.
# What is sensitivity analysis?
# Sensitivity analysis involves changing input parameters within certain ranges or constraints and observing how these changes to input affect the output. 
# The product of this analysis technique is a set of input parameters are most influential on the model's predictions.
# How to perform sensitivity analysis?
# 1.Perturb Input: Slightly change one input parameter while keeping others constant.
# 2. Observe Output Change: Measure how the output changes in response to this perturbation.
# 3. Repeat for Each Input: Perform this for each input parameter to understand their individual impact.

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

csvFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Train.csv"
testFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Test.csv"
midFile = "C:/Users/S2996310/Subaru/avg of mid points.csv"
testmidFile = "C:/Users/S2996310/Subaru/test avg of mid points.csv"

Adam_learning_rate = 0.0066
Nepochs = 2000
Nneurons_Dense = 320
activation = 'relu'
batch = 32

# Load the model
model = load_model('model')  

input_dim = 6
base_input = np.random.normal(size=(1, input_dim))  # Base input
sensitivity = np.zeros(input_dim)

epsilon = 0.01  # Small perturbation (https://en.wikipedia.org/wiki/Sensitivity_analysis)

for i in range(input_dim):
    perturbed_input = np.copy(base_input)
    perturbed_input[0, i] += epsilon  # Perturb i-th input
    
    original_output = model.predict(base_input)
    new_output = model.predict(perturbed_input)

    sensitivity[i] = (new_output - original_output) / epsilon

print("Sensitivity of each input dimension:", sensitivity)
