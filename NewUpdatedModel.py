import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5000)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_model = lgb.LGBMRegressor()
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

y_pred_test = model.predict(X_test_scaled)

mse_ex = mean_squared_error(y_test['Ex (Excitation Energy in keV)'], y_pred_test[:, 0])
r2_ex = r2_score(y_test['Ex (Excitation Energy in keV)'], y_pred_test[:, 0])
mse_be3 = mean_squared_error(y_test['B(E3) (Adopted Transition Probability in e^2b^3)'], y_pred_test[:, 1])
r2_be3 = r2_score(y_test['B(E3) (Adopted Transition Probability in e^2b^3)'], y_pred_test[:, 1])

print(f"\nTest Set - Excitation Energy: MSE: {mse_ex:.4f}, R²: {r2_ex:.4f}")
print(f"Test Set - Transition Probability: MSE: {mse_be3:.4f}, R²: {r2_be3:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test['Ex (Excitation Energy in keV)'], y_pred_test[:, 0], alpha=0.7)
plt.xlabel("Actual Excitation Energy (keV)")
plt.ylabel("Predicted Excitation Energy (keV)")
plt.title("Actual vs. Predicted Excitation Energy")
plt.plot([y_test['Ex (Excitation Energy in keV)'].min(), y_test['Ex (Excitation Energy in keV)'].max()],
         [y_test['Ex (Excitation Energy in keV)'].min(), y_test['Ex (Excitation Energy in keV)'].max()],
         'r--')
plt.subplot(1, 2, 2)
plt.scatter(y_test['B(E3) (Adopted Transition Probability in e^2b^3)'], y_pred_test[:, 1], alpha=0.7)
plt.xlabel("Actual Transition Probability (e^2b^3)")
plt.ylabel("Predicted Transition Probability (e^2b^3)")
plt.title("Actual vs. Predicted Transition Probability")
plt.plot([y_test['B(E3) (Adopted Transition Probability in e^2b^3)'].min(), y_test['B(E3) (Adopted Transition Probability in e^2b^3)'].max()],
         [y_test['B(E3) (Adopted Transition Probability in e^2b^3)'].min(), y_test['B(E3) (Adopted Transition Probability in e^2b^3)'].max()],
         'r--')
plt.tight_layout()
plt.show()

explainer = shap.Explainer(model.estimators_[0], X_train_scaled)
shap_values = explainer(X_train_scaled, check_additivity=False)
shap.summary_plot(shap_values, X_train_scaled, feature_names=featlist)


