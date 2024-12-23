import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\aravk\Downloads\Processed_Nuclear_Physics_Data.xlsx"  # Update path
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
    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }

    # Create and train the XGBoost model
    model = xgb.XGBRegressor(**params, random_state=42, eval_metric='rmse')

    # Manual cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=0)
        y_pred = model.predict(X_fold_val)
        r2_scores.append(r2_score(y_fold_val, y_pred))

    return np.mean(r2_scores)

# Perform Bayesian Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

print(f"Best R2: {study.best_value}")
print(f"Best Parameters: {study.best_params}")

# Train the best model with optimal parameters
best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params, random_state=42, eval_metric='rmse')
best_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_best = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)

print(f"Test MSE: {mse}")
print(f"Test R2: {r2}")

# Plot: Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, label='Predictions', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual B(E3)")
plt.ylabel("Predicted B(E3)")
plt.title("XGBoost: Actual vs Predicted B(E3)")
plt.legend()
plt.grid(True)
plt.show()
