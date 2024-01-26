import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt

# Load your data from CSV
data = pd.read_csv(r'exports\CSV_exports\231208_4xCB-loading_KBrTransmission_ambient-cure_consolidated_PV-k-values.csv')
k = data['k']
k_error = data['k_error']
loading = data['loading']

# Define a logarithmic model function
def log_model(params, x):
    a = params['a']
    b = params['b']
    c = params['c']
    return a * np.log(x + b) + c

# Create a set of Parameters
params = lmfit.Parameters()
params.add('a', value=3.83e-3)
params.add('b', value=4.88e-4, min=0)  # b should be positive to avoid log(0)
params.add('c', value=8.55e-2)

# Define an objective function for the fit, including errors in k
def objective(params, x, data, error):
    model = log_model(params, x)
    return (data - model) / error

# Perform the fit
minimizer = lmfit.Minimizer(objective, params, fcn_args=(loading, k, k_error))
result = minimizer.minimize()

# Print the fit report
print(lmfit.fit_report(result))

# Generate values for the fitted curve
loading_fitted = np.linspace(min(loading), max(loading), 500)
k_fitted = log_model(result.params, loading_fitted)

# Plotting the data with error bars
plt.errorbar(loading, k, yerr=k_error, fmt='o', label='Data with error', ecolor='red', capsize=5)

# Plotting the fitted curve
plt.plot(loading_fitted, k_fitted, '-', label='Fitted curve', color='blue')

# Adding labels and legend
plt.xlabel('Loading')
plt.ylabel('k')
plt.legend()

# Show the plot
plt.show()