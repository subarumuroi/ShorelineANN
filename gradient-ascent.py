csvFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Train.csv"
testFile = "C:/Users/S2996310/Subaru/AllRuns_SingleTime_Test.csv"
midFile = "C:/Users/S2996310/Subaru/avg of mid points.csv"
testmidFile = "C:/Users/S2996310/Subaru/test avg of mid points.csv"

Adam_learning_rate = 0.0066
Nepochs = 2000
Nneurons_Dense = 320
activation = 'relu'
batch = 32

verbose = 1 # 1 => display model training/prediction progress data to console or 0 => don't

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

import tensorflow as tf
import numpy as np


#compute skill metric
def getSkillMetrics(actual_values, predicted_values):
    
    metrics_list = [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanAbsolutePercentageError(),
        tf.keras.metrics.RootMeanSquaredError(),
    ]
    
    # Dictionary to store computed metrics
    computed_metrics = {}
    
    # Loop over metrics and compute each metric
    for metric in metrics_list:
        metric.update_state(actual_values, predicted_values)
        computed_metric_value = metric.result().numpy()
        computed_metrics[metric.name] = computed_metric_value
    
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(list(computed_metrics.items()), columns=['Metric', 'Value'])
    
    return metrics_df



print('  Loading data ...')
#create dataframes
df = pd.read_csv(csvFile) #main training dataset and stats file
dftest = pd.read_csv(testFile) #main testing dataset and stats file
dfmid = pd.read_csv(midFile) #average midvalues of shoreline for training
dftestmid = pd.read_csv(testmidFile) #avg mid of shoreline for testing

#remove unwatned row labels in first column
columns = df.columns # get dataframe column header names
columns = columns[1:244] #remove the header from the first column 
columns_test = dftest.columns # get dataframe column header names
columns_test = columns_test[1:40] #remove the header from the first column 


# sample the df and convert to numpy array
'''
the "df[columns]" selects the columns specified in the columns array from above;
the ".iloc[3:8]" selects rows 2 to 8 which correspond to relevant SBW and wave parameters in the csv file
the ".values.astype(float)" converts the df to a numpy array of floats
'''
X_train = df[columns].iloc[2:8].values.astype(float).T  


'''
Below you can adjust which values to train on (number of midpoint values averaged)
'''
#Y_train = dfmid[columns].iloc[0].values.astype(float).T #absolute mid point
#Y_train = dfmid[columns].iloc[5].values.astype(float).T # avg of mid 3
#Y_train = dfmid[columns].iloc[12].values.astype(float).T # avg of mid 5
Y_train = dfmid[columns].iloc[21].values.astype(float).T # avg of mid 7


X_test = dftest[columns_test].iloc[2:8].values.astype(float).T  

#Y_test = dftest[columns_test].iloc[15].values.astype(float) # row 12 is the max shoreline row
Y_test = dftestmid[columns_test].iloc[5].values.astype(float).T # entire shoreline output



print('  Training model (be patient!) ...')
#create nn model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
    tf.keras.layers.Dense(1)  #output layer with 1var
])
#optimizer
opt = tf.keras.optimizers.legacy.Adam(learning_rate = Adam_learning_rate, clipnorm = 1)
#compile the model
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

fit=model.fit(X_train, Y_train, epochs=Nepochs, batch_size=batch, verbose=verbose)

# loss plot
plt.figure()
plt.plot(fit.history['loss'])
plt.title('Model Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# compute model predictions - training
Y_pred = model.predict(X_train, verbose=verbose)

# plot training performance
plt.figure()
plt.scatter(Y_train, Y_pred, color='blue')
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values: Training Data')

# Annotate the plot with computed metrics
metrics_df = getSkillMetrics(Y_train, Y_pred)
for index, row in metrics_df.iterrows():
    plt.annotate(f"{row['Metric']}: {row['Value']:.4f}", (0.05, 0.95 - 0.05 * index),
                 xycoords='axes fraction', fontsize=10)   
plt.show()

print('  Applying model to test data ...')
Y_pred = model.predict(X_test, verbose=verbose)

# Plot predictions vs actual values
plt.figure()
plt.scatter(Y_test, Y_pred, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values: Test Data')

# Annotate the plot with computed metrics
metrics_df = getSkillMetrics(Y_test, Y_pred)
for index, row in metrics_df.iterrows():
    plt.annotate(f"{row['Metric']}: {row['Value']:.4f}", (0.05, 0.95 - 0.05 * index),
                 xycoords='axes fraction', fontsize=10)   
plt.show()

model.save('model') 

# k-means

# df = working dataframe

from sklearn.model_selection import KFold

# define k-fold measures
k = 5
kf = KFold(n_splits=k)

# iterate over folds for training and validation
scores = []
count = 1
for train_index, val_index in kf.split(X_train, Y_train):
    print(f'Fold:{count}, Train set: {len(train_index)}, Test set:{len(val_index)}')
    count += 1

# build model and run

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]
    
    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(6,)),  #input layer with 6var
        tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
        tf.keras.layers.Dense(Nneurons_Dense, activation=activation),
        tf.keras.layers.Dense(1)  #output layer with 1var
    ])
    
    # Compile & fit
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    model.fit(X_train_fold, Y_train_fold, epochs = 2000, batch_size = 32)

    score = model.evaluate(X_val_fold, Y_val_fold)
    scores.append(score)
    
 
# print scores
mean_score0 = np.mean(scores, axis = 0)
mean_score1 = np.mean(scores, axis = 1)
mean_score = np.mean(scores)
    
mean_score
mean_score1
mean_score