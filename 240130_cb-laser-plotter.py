import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Data
k = np.array([0.05287584996876471, 0.06835624989041476, 0.0940549075365208])
k_error = np.array([0.03823427569941955, 0.023102367686801555, 0.03771380236238723])
amount = np.array([0.000001, 0.0001, 0.1])

# Define the linear model function
def linear_model(x, a, b):
    return a * x + b

# Inverse of the square of the errors as weights for the fit
weights = 1 / k_error**2

# Perform the weighted linear fit
popt_weighted, pcov_weighted = curve_fit(linear_model, np.log10(amount), k, sigma=k_error, absolute_sigma=True)
perr_weighted = np.sqrt(np.diag(pcov_weighted))  # Standard deviation errors on the weighted parameters

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(np.log10(amount), k, yerr=k_error, fmt='o', label='Data with Error Bars', color='#023e8a', capsize=5, markersize=10)
plt.plot(np.log10(amount), linear_model(np.log10(amount), *popt_weighted), label=f'Weighted Linear Fit: y = {popt_weighted[0]:.5e}x + {popt_weighted[1]:.5e}', color = '#0077b6', linewidth=5)

plt.xlabel('log(loading)', fontsize=40)
plt.ylabel('k', fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid(False)

# error
print(f"Slope: {popt_weighted[0]:.5e}")
print(f"Slope Error: {perr_weighted[0]:.5e}")
print(f"Difference: {perr_weighted[0] - popt_weighted[0]:.5e}")

plt.show()