import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

file_path = r"C:\Users\aravk\Downloads\Nuclear_Physics_Data_Set_Updated_FINALFINAL.xlsx"
data = pd.read_excel(file_path)

targetlist = ['Ex (Excitation Energy in keV)', 'B(E3) (Adopted Transition Probability in e^2b^3)']
featlist = ['ZZ (Proton Number)', 
            'NN (Neutron Number)', 
            'A = Z + N (Mass Number)',
            'Average β3 (Deformation Parameter)',
            'protonSeparationEnergy(keV)', 
            'neutronSeparationEnergy(keV)',
            'twoProtonSeparationEnergy(keV)', 
            'twoNeutronSeparationEnergy(keV)',
            'Casten Factor', 
            'Binding Energy']

X = data[featlist].copy()
y = data[targetlist].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=677)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_ex = y_train['Ex (Excitation Energy in keV)'].values
y_train_be3 = y_train['B(E3) (Adopted Transition Probability in e^2b^3)'].values
y_test_ex = y_test['Ex (Excitation Energy in keV)'].values
y_test_be3 = y_test['B(E3) (Adopted Transition Probability in e^2b^3)'].values


ex_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
be3_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)

ex_model = GaussianProcessRegressor(kernel=ex_kernel, n_restarts_optimizer=10, random_state=677)
be3_model = GaussianProcessRegressor(kernel=be3_kernel, n_restarts_optimizer=10, random_state=677)

ex_model.fit(X_train_scaled, y_train_ex)
be3_model.fit(X_train_scaled, y_train_be3)

y_pred_ex = ex_model.predict(X_test_scaled)
y_pred_be3 = be3_model.predict(X_test_scaled)

mse_ex = mean_squared_error(y_test_ex, y_pred_ex)
r2_ex = r2_score(y_test_ex, y_pred_ex)
mse_be3 = mean_squared_error(y_test_be3, y_pred_be3)
r2_be3 = r2_score(y_test_be3, y_pred_be3)

print(f"Excitation Energy: rMSE: {math.sqrt(mse_ex):.4f}, R²: {r2_ex:.4f}")
print(f"Transition Probability: rMSE: {math.sqrt(mse_be3):.4f}, R²: {r2_be3:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_ex, y_pred_ex, alpha=0.7)
plt.xlabel("Actual Excitation Energy (keV)")
plt.ylabel("Predicted Excitation Energy (keV)")
plt.title(f"Actual vs. Predicted Excitation Energy\nR² = {r2_ex:.4f}")
plt.plot([min(y_test_ex), max(y_test_ex)], [min(y_test_ex), max(y_test_ex)], 'r--')

plt.subplot(1, 2, 2)
plt.scatter(y_test_be3, y_pred_be3, alpha=0.7)
plt.xlabel("Actual Transition Probability (e^2b^3)")
plt.ylabel("Predicted Transition Probability (e^2b^3)")
plt.title(f"Actual vs. Predicted Transition Probability\nR² = {r2_be3:.4f}")
plt.plot([min(y_test_be3), max(y_test_be3)], [min(y_test_be3), max(y_test_be3)], 'r--')

plt.tight_layout()
plt.show()
