import pandas as pd
import matplotlib.pyplot as plt

# Specify the file path
file_path = "/Users/aravk/Desktop/Updated_Nuclear_Physics_Data.xlsx"

try:
    # Load the Excel file
    data = pd.read_excel(file_path)

    # Extract the required columns
    proton_number = data["ZZ (Proton Number)"]
    excitation_energy = data["Ex (Excitation Energy in keV)"]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(proton_number, excitation_energy, color="blue", alpha=0.7, edgecolors="k")
    plt.title("Excitation Energy vs Proton Number", fontsize=14)
    plt.xlabel("Proton Number (ZZ)", fontsize=12)
    plt.ylabel("Excitation Energy (keV)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

except FileNotFoundError:
    print(f"The file at {file_path} was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
