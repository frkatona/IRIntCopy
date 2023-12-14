import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit import Model

# First order kinetic decay function
def first_order_kinetic_decay(x, A_0, k, C):
    return A_0 * np.exp(-k * x) + C

# Load the CSV file
file_path = r'exports\CSV_exports\231208_4xCB-loading_KBrTransmission_ambient-cure_consolidated_PV-Amplitudes.csv'  # Replace with your CSV file path
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
        result = model.fit(subset['Amplitude'], params, x=subset['Time Value'])

        if result.success and result.chisqr < lowest_error:
            best_popt = [result.params['A_0'].value, result.params['k'].value, result.params['C'].value]
            lowest_error = result.chisqr

    if best_popt is not None:
        fit_statistics.append(f"Loading {loading}%: A_t = {best_popt[0]:.4f} * e^(-{best_popt[1]:.4f}*t) + {best_popt[2]:.4f}, Error: {lowest_error:.4e}")
        k_values[loading] = {'k': best_popt[1], 'k_error': result.params['k'].stderr}  # Storing the k value and its error
        x_range = np.linspace(subset['Time Value'].min(), subset['Time Value'].max(), 500)
        plt.plot(x_range, first_order_kinetic_decay(x_range, *best_popt), color=colors[i])

plt.title('Scatter Plot with First Order Kinetic Decay Fit for Each Agent Loading')
plt.xlabel('time /h')
plt.ylabel('Amplitude')
plt.legend(title='Agent Loading (wt%)', loc='best')

# Print the fit equations and their statistics
for stat in fit_statistics:
    print(stat)

# Scatter plot for k values
plt.figure(figsize=(10, 6))
plt.scatter(k_values.keys(), [v['k'] for v in k_values.values()], marker='o', s=100)
plt.errorbar(k_values.keys(), [v['k'] for v in k_values.values()], yerr=[v['k_error'] for v in k_values.values()], fmt='none')
plt.xscale('log')  # Setting the x-axis to logarithmic scale
plt.xlabel('Agent Loading (wt%)')
plt.ylabel('k Value')
plt.title('Scatter Plot of k Values for Each Agent Loading (Logarithmic Scale)')
plt.grid(True)

# Scatter plot for k values
plt.figure(figsize=(10, 6))
x_coordinates = np.arange(len(k_values))  # Generate equally spaced x-coordinates
plt.bar(x_coordinates, [v['k'] for v in k_values.values()], color='blue')
plt.errorbar(x_coordinates, [v['k'] for v in k_values.values()], yerr=[v['k_error'] for v in k_values.values()], fmt='none', color='black')
plt.xticks(x_coordinates, k_values.keys())  # Set the x-axis labels to the agent loadings
plt.xlabel('Agent Loading (wt%)')
plt.ylabel('k Value')
plt.title('Bar Chart of k Values for Each Agent Loading')
plt.grid(True)
plt.show()