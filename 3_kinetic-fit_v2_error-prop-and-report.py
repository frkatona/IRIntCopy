import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit import Model

# First order kinetic decay function
def first_order_kinetic_decay(x, A_0, k, C):
    return A_0 * np.exp(-k * x) + C

# Load the CSV file
file_path = 'exports/CSV_exports/231208_4xCB-loading_KBrTransmission_ambient-cure_consolidated_PV-Amplitudes.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Preparing the scatter plot
plt.figure(figsize=(12, 8))
sns.set(style="ticks")

unique_loadings = sorted(data['Agent Loading'].unique())
colors = sns.color_palette("Blues", len(unique_loadings))

initial_guesses = [(1, 0.1, 0), (0.1, 0.01, 0), (0.5, 0.05, 0.01)]

fit_statistics = []
k_values = {}  # Dictionary to store k values

# Scatter plot and curve fitting
for i, loading in enumerate(unique_loadings):
    subset = data[data['Agent Loading'] == loading]
    sns.scatterplot(x=subset['Time Value'], y=subset['Amplitude'], color=colors[i], label=f'{loading}%', marker='o', s=100)

    # Add error bars
    plt.errorbar(subset['Time Value'], subset['Amplitude'], yerr=subset['Amplitude Error'], fmt='none', color=colors[i])

    best_popt = None
    lowest_error = float('inf')

    for guess in initial_guesses:
        model = Model(first_order_kinetic_decay)
        params = model.make_params(A_0=guess[0], k=guess[1], C=guess[2])
        result = model.fit(subset['Amplitude'], params, x=subset['Time Value'], weights=1 / subset['Amplitude Error'])

        if result.success and result.chisqr < lowest_error:
            best_popt = [result.params['A_0'].value, result.params['k'].value, result.params['C'].value]
            lowest_error = result.chisqr
            best_result = result  # Save the best fit result for later use

    if best_popt is not None:
        # Storing the k value and its error
        k_values[loading] = {'k': best_popt[1], 'k_error': best_result.params['k'].stderr}  
        
        # Printing the parameter values and errors
        print(f"Fit Report for Loading {loading}%:")
        for name, param in best_result.params.items():
            print(f'{name}: {param.value} ± {param.stderr}')

        # Plotting the best fit
        x_range = np.linspace(subset['Time Value'].min(), subset['Time Value'].max(), 500)
        plt.plot(x_range, first_order_kinetic_decay(x_range, *best_popt), color=colors[i])

plt.title('Scatter Plot with First Order Kinetic Decay Fit for Each Agent Loading')
plt.xlabel('time /h')
plt.ylabel('Amplitude')
plt.legend(title='Agent Loading (wt%)', loc='best')

# Define the exponential model
def exp_model(x, A, B, C):
    return A * np.exp(B * x) + C

# Convert loading and k_values into arrays for fitting
loadings_array = np.array(list(k_values.keys()), dtype=float)
k_values_array = np.array([v['k'] for v in k_values.values()])
k_errors_array = np.array([v['k_error'] for v in k_values.values()])

# Create a model and parameters
exp_model = Model(exp_model)
params = exp_model.make_params(A=1, B=0.1, C=0)

# Perform the fit
result = exp_model.fit(k_values_array, params, x=loadings_array, weights=1/k_errors_array)

# Plotting the fit
plt.figure(figsize=(12, 8))
plt.errorbar(loadings_array, k_values_array, yerr=k_errors_array, fmt='o', label='Data with error')
plt.plot(loadings_array, result.best_fit, label='Fitted curve', color='red')
plt.title('Exponential Fit of k vs. Loading')
plt.xlabel('Loading')
plt.ylabel('k')
plt.legend()

# Print the fit report
print("Fit Report for k vs. Loading:")
print(result.fit_report())

plt.show()