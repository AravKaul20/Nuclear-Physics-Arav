import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5000)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_lgbm = lgb.LGBMRegressor(
    random_state=5000
)

multi_lgbm = MultiOutputRegressor(base_lgbm)
multi_lgbm.fit(X_train_scaled, y_train)

y_pred = multi_lgbm.predict(X_test_scaled)

y_test_values = y_test.values
mse_ex = mean_squared_error(y_test_values[:, 0], y_pred[:, 0])
r2_ex = r2_score(y_test_values[:, 0], y_pred[:, 0])
mse_be3 = mean_squared_error(y_test_values[:, 1], y_pred[:, 1])
r2_be3 = r2_score(y_test_values[:, 1], y_pred[:, 1])

print(f"Excitation Energy: rMSE: {math.sqrt(mse_ex):.4f}, R²: {r2_ex:.4f}")
print(f"Transition Probability: rMSE: {math.sqrt(mse_be3):.4f}, R²: {r2_be3:.4f}")
print(f"Average R²: {(r2_ex + r2_be3) / 2:.4f}")
