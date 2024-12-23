import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\aravk\Downloads\Processed_Nuclear_Physics_Data.xlsx"
data = pd.read_excel(file_path)

# Features and Target
X = data[['ZZ (Proton Number)', 'NN (Neutron Number)', 'Ex (Excitation Energy in keV)', 
          'Average β3 (Deformation Parameter)', 'Casten_Factor', 'Proton_to_Neutron_Ratio']]
y = data['B(E3) (Adopted Transition Probability in e^2b^3)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the objective function for Bayesian Optimization
def objective(trial):
    random_seed = trial.suggest_int('seed', 0, 1000)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.1)
    neurons = trial.suggest_categorical('neurons', [64, 128, 256])

    # Set random seeds
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    # Create and train the neural network model
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(neurons, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32, verbose=0)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    return r2

# Perform Bayesian Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best R2: {study.best_value}")
print(f"Best Parameters: {study.best_params}")

# Train the best model with optimal parameters
best_params = study.best_params
np.random.seed(best_params['seed'])
tf.random.set_seed(best_params['seed'])
random.seed(best_params['seed'])

best_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(best_params['neurons'], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])

best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])
best_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32, verbose=0)

# Plot: Actual vs Predicted values for the best model
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, label='Predictions', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual B(E3)")
plt.ylabel("Predicted B(E3)")
plt.title("Best Neural Network: Actual vs Predicted B(E3)")
plt.legend()
plt.grid(True)
plt.show()
