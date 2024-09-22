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



#NN

import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model_NN_best.h5')

# Load normalization parameters
normalization_params = np.load('normalization_params.npy', allow_pickle=True).item()

# Define parameter ranges
ParameterRanges = {'H': [0.5, 2.5],
                   'T': [7.0, 13.0],
                   'Dir': [180.0, 180.0],
                   'Dist': [150.0, 450.0],
                   'L': [100.0, 200.0],
                   'W': [25.0, 75.0]}

# Map normalization parameters to parameter ranges
norm_to_range = {}

for i, (param, _) in enumerate(ParameterRanges.items()):
    norm_to_range[param] = {'scaling_mean': normalization_params['scaling_mean'][i],
                            'scaling_std': normalization_params['scaling_std'][i]}


input_dim = 6
output_dim = 51  # Number of output values
epsilon = 0.1  # Small perturbation (https://en.wikipedia.org/wiki/Sensitivity_analysis)
num_runs = 100  # Number of sensitivity analyses to run
sensitivity_results = []

for run in range(num_runs):
    sensitivity = np.zeros((input_dim, output_dim))
    base_input = np.zeros((1, input_dim))

    # Generate random base input within specified ranges
    for i, (param, (min_val, max_val)) in enumerate(ParameterRanges.items()):
        base_input[0, i] = np.random.uniform(min_val, max_val)

    
    for i in range(input_dim):
        perturbed_input = np.copy(base_input)
        perturbed_input[0, i] += epsilon  # Perturb i-th input

        # Access normalization parameters using index i
        perturbed_input_normalized = (perturbed_input - normalization_params['scaling_mean'][i]) / normalization_params['scaling_std'][i]
        base_input_normalized = (base_input - normalization_params['scaling_mean'][i]) / normalization_params['scaling_std'][i]

        original_output = model.predict(base_input_normalized)
        new_output = model.predict(perturbed_input_normalized)

        sensitivity[i] = (new_output - original_output) / epsilon

    sensitivity_results.append(sensitivity)
    
# Average the sensitivity results over all runs
average_sensitivity = np.mean(sensitivity_results, axis=0)

print("Average sensitivity of each input dimension over", num_runs, "runs:")
print(average_sensitivity)

# Compute summary statistics for each input dimension's sensitivity values
mean_sensitivity = np.mean(average_sensitivity, axis=1)
std_sensitivity = np.std(average_sensitivity, axis=1)
min_sensitivity = np.min(average_sensitivity, axis=1)
max_sensitivity = np.max(average_sensitivity, axis=1)

# Print summary statistics for each input dimension
for i, param in enumerate(ParameterRanges.keys()):
    print(f"Parameter: {param}")
    print(f"Mean sensitivity: {mean_sensitivity[i]}")
    print(f"Standard deviation: {std_sensitivity[i]}")
    print(f"Min sensitivity: {min_sensitivity[i]}")
    print(f"Max sensitivity: {max_sensitivity[i]}")
    print()

import matplotlib.pyplot as plt

# Define alongshore distances
alongshore_distances = [1544.811279, 1508.018921, 1471.22644, 1434.43396, 1397.641479,
                        1360.848999, 1324.056641, 1287.26416, 1250.47168, 1213.679199, 1176.886841,
                        1140.09436, 1103.30188, 1066.509399, 1029.717041, 992.9244995, 956.1320801,
                        919.3395996, 882.5471802, 845.7546997, 808.9622803, 772.1697998, 735.3773804,
                        698.5848999, 661.7924805, 625, 588.2075195, 551.4151001, 514.6226196, 477.8302002,
                        441.0377502, 404.2452698, 367.4528198, 330.6603699, 293.8679199, 257.07547,
                        220.28302, 183.4905701, 146.6981201, 109.9056625, 73.11320496, 36.320755,
                        -0.471698105, -37.26415253, -74.05660248, -110.8490601, -147.64151, -184.43396,
                        -221.2264099, -258.0188599, -294.8113098]

# Define input dimension names
input_dim_names = list(ParameterRanges.keys())

# Plot sensitivity values for each input dimension
plt.figure(figsize=(10, 6))

for i, param in enumerate(input_dim_names):
    plt.plot(alongshore_distances, average_sensitivity[i], label=param)

plt.xlabel('Alongshore Distance')
plt.ylabel('Average Sensitivity')
plt.title('Average Sensitivity of Each Input Dimension')
plt.legend()
plt.grid(True)
plt.show()


### NN with no normalization seems to work best so far.
'''
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model_NN_best.h5')  

# Define parameter ranges
ParameterRanges = {'H': [0.5, 2.5],
                   'T': [7.0, 13.0],
                   'Dir': [180.0, 180.0],
                   'Dist': [150.0, 450.0],
                   'L': [100.0, 200.0],
                   'W': [25.0, 75.0]}

input_dim = 6
output_dim = 51  # Number of output values
epsilon = 0.01  # Small perturbation (https://en.wikipedia.org/wiki/Sensitivity_analysis)
num_runs = 100  # Number of sensitivity analyses to run
sensitivity_results = []

for run in range(num_runs):
    sensitivity = np.zeros((input_dim, output_dim))
    base_input = np.zeros((1, input_dim))

    # Generate random base input within specified ranges
    for i, (param, (min_val, max_val)) in enumerate(ParameterRanges.items()):
        base_input[0, i] = np.random.uniform(min_val, max_val)

    for i in range(input_dim):
        perturbed_input = np.copy(base_input)
        perturbed_input[0, i] += epsilon  # Perturb i-th input
        
        original_output = model.predict(base_input)
        new_output = model.predict(perturbed_input)

        sensitivity[i] = (new_output - original_output) / epsilon

    sensitivity_results.append(sensitivity)

# Average the sensitivity results over all runs
average_sensitivity = np.mean(sensitivity_results, axis=0)

print("Average sensitivity of each input dimension over", num_runs, "runs:")
print(average_sensitivity)

# Compute summary statistics for each input dimension's sensitivity values
mean_sensitivity = np.mean(average_sensitivity, axis=1)
std_sensitivity = np.std(average_sensitivity, axis=1)
min_sensitivity = np.min(average_sensitivity, axis=1)
max_sensitivity = np.max(average_sensitivity, axis=1)

# Print summary statistics for each input dimension
for i, param in enumerate(ParameterRanges.keys()):
    print(f"Parameter: {param}")
    print(f"Mean sensitivity: {mean_sensitivity[i]}")
    print(f"Standard deviation: {std_sensitivity[i]}")
    print(f"Min sensitivity: {min_sensitivity[i]}")
    print(f"Max sensitivity: {max_sensitivity[i]}")
    print()


import matplotlib.pyplot as plt

# Define alongshore distances
alongshore_distances = [1544.811279, 1508.018921, 1471.22644, 1434.43396, 1397.641479,
                        1360.848999, 1324.056641, 1287.26416, 1250.47168, 1213.679199, 1176.886841,
                        1140.09436, 1103.30188, 1066.509399, 1029.717041, 992.9244995, 956.1320801,
                        919.3395996, 882.5471802, 845.7546997, 808.9622803, 772.1697998, 735.3773804,
                        698.5848999, 661.7924805, 625, 588.2075195, 551.4151001, 514.6226196, 477.8302002,
                        441.0377502, 404.2452698, 367.4528198, 330.6603699, 293.8679199, 257.07547,
                        220.28302, 183.4905701, 146.6981201, 109.9056625, 73.11320496, 36.320755,
                        -0.471698105, -37.26415253, -74.05660248, -110.8490601, -147.64151, -184.43396,
                        -221.2264099, -258.0188599, -294.8113098]

# Define input dimension names
input_dim_names = list(ParameterRanges.keys())

# Plot sensitivity values for each input dimension
plt.figure(figsize=(10, 6))

for i, param in enumerate(input_dim_names):
    plt.plot(alongshore_distances, average_sensitivity[i], label=param)

plt.xlabel('Alongshore Distance')
plt.ylabel('Average Sensitivity')
plt.title('Average Sensitivity of Each Input Dimension')
plt.legend()
plt.grid(True)
plt.show()

'''


####GPR 
'''
import numpy as np
import joblib

# Load the GPR model
model = joblib.load('gpr_model_used_for_MC.pkl')

# Load normalization parameters
normalization_params = np.load('normalization_params.npy', allow_pickle=True).item()


# Define parameter ranges
ParameterRanges = {'H': [0.5, 2.5],
                   'T': [7.0, 13.0],
                   'Dir': [180.0, 180.0],
                   'Dist': [150.0, 450.0],
                   'L': [100.0, 200.0],
                   'W': [25.0, 75.0]}

input_dim = 6
output_dim = 51  # Number of output values
epsilon = .00001  # Small perturbation (https://en.wikipedia.org/wiki/Sensitivity_analysis)
num_runs = 100  # Number of sensitivity analyses to run
sensitivity_results = []

for run in range(num_runs):
    sensitivity = np.zeros((input_dim, output_dim))
    base_input = np.zeros((1, input_dim))

    # Generate random base input within specified ranges
    for i, (param, (min_val, max_val)) in enumerate(ParameterRanges.items()):
        base_input[0, i] = np.random.uniform(min_val, max_val)

    for i in range(input_dim):
        perturbed_input = np.copy(base_input)
        perturbed_input[0, i] += epsilon  # Perturb i-th input
        
        original_output = gpr_model.predict(base_input)
        new_output = gpr_model.predict(perturbed_input)

        sensitivity[i] = (new_output - original_output) / epsilon

    sensitivity_results.append(sensitivity)

    
# Average the sensitivity results over all runs
average_sensitivity = np.mean(sensitivity_results, axis=0)

print("Average sensitivity of each input dimension over", num_runs, "runs:")
print(average_sensitivity)

# Compute summary statistics for each input dimension's sensitivity values
mean_sensitivity = np.mean(average_sensitivity, axis=1)
std_sensitivity = np.std(average_sensitivity, axis=1)
min_sensitivity = np.min(average_sensitivity, axis=1)
max_sensitivity = np.max(average_sensitivity, axis=1)

# Print summary statistics for each input dimension
for i, param in enumerate(ParameterRanges.keys()):
    print(f"Parameter: {param}")
    print(f"Mean sensitivity: {mean_sensitivity[i]}")
    print(f"Standard deviation: {std_sensitivity[i]}")
    print(f"Min sensitivity: {min_sensitivity[i]}")
    print(f"Max sensitivity: {max_sensitivity[i]}")
    print()

import matplotlib.pyplot as plt

# Define alongshore distances
alongshore_distances = [1544.811279, 1508.018921, 1471.22644, 1434.43396, 1397.641479,
                        1360.848999, 1324.056641, 1287.26416, 1250.47168, 1213.679199, 1176.886841,
                        1140.09436, 1103.30188, 1066.509399, 1029.717041, 992.9244995, 956.1320801,
                        919.3395996, 882.5471802, 845.7546997, 808.9622803, 772.1697998, 735.3773804,
                        698.5848999, 661.7924805, 625, 588.2075195, 551.4151001, 514.6226196, 477.8302002,
                        441.0377502, 404.2452698, 367.4528198, 330.6603699, 293.8679199, 257.07547,
                        220.28302, 183.4905701, 146.6981201, 109.9056625, 73.11320496, 36.320755,
                        -0.471698105, -37.26415253, -74.05660248, -110.8490601, -147.64151, -184.43396,
                        -221.2264099, -258.0188599, -294.8113098]

# Define input dimension names
input_dim_names = list(ParameterRanges.keys())

# Plot sensitivity values for each input dimension
plt.figure(figsize=(10, 6))

for i, param in enumerate(input_dim_names):
    plt.plot(alongshore_distances, average_sensitivity[i], label=param)

plt.xlabel('Alongshore Distance')
plt.ylabel('Average Sensitivity')
plt.title('Average Sensitivity of Each Input Dimension')
plt.legend()
plt.grid(True)
plt.show()
'''
