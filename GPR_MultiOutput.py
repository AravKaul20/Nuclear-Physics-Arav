import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor

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

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0)
base_gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=677
)

multi_gpr = MultiOutputRegressor(base_gpr)
multi_gpr.fit(X_train_scaled, y_train)

y_pred = multi_gpr.predict(X_test_scaled)

y_test_values = y_test.values
mse_ex = mean_squared_error(y_test_values[:, 0], y_pred[:, 0])
r2_ex = r2_score(y_test_values[:, 0], y_pred[:, 0])
mse_be3 = mean_squared_error(y_test_values[:, 1], y_pred[:, 1])
r2_be3 = r2_score(y_test_values[:, 1], y_pred[:, 1])

print(f"Excitation Energy: rMSE: {math.sqrt(mse_ex):.4f}, R²: {r2_ex:.4f}")
print(f"Transition Probability: rMSE: {math.sqrt(mse_be3):.4f}, R²: {r2_be3:.4f}")
