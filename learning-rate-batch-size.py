import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.datasets import make_regression

# Random/testing dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create model
def create_model(learning_rate=0.01, batch_size=32):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Wrap the model for scikit-learn
model = KerasRegressor(build_fn=create_model, learning_rate=0.001, verbose=0)

# Hyperparameters to search, including learning rate and batch size
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50], # Random choice, can be adjusted to experiment
}

# Grid search learning rate and batch size
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_result = grid_search.fit(X_train, y_train)

# Pull out best hyperparams
best_learning_rate = grid_result.best_params_['learning_rate']
best_batch_size = grid_result.best_params_['batch_size']
best_model = grid_result.best_estimator_

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
    X_train, y_train,
    epochs=50,  # Adjust the number of epochs as needed
    batch_size=best_batch_size,
    validation_data=(X_test, y_test),  # Use the test set for validation
    callbacks=[lr_callback, early_stopping_callback]
)

# Plot the learning rate range test results (loss vs. learning rate)
learning_rates = [10 ** i for i in range(-5, 0)]  # A likely range of learning rates (would not expect 10e-6 or lower)
losses = []

for lr in learning_rates:
    model = create_model(learning_rate=lr, batch_size=best_batch_size)
    history = model.fit(
        X_train, y_train,
        epochs=10,  # I have used a small epoch size. Your speedy Mac could probably handle more :)
        batch_size=best_batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )
    losses.append(history.history['val_loss'][-1])

plt.semilogx(learning_rates, losses, marker='o')
plt.title('Learning Rate Range Test')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.show()

# Final eval, printing best params
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Batch Size: {best_batch_size}')
print(f'Mean Squared Error on Test Data: {mse}')
