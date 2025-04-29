import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit import Model

# use lmfit again for the k fit and tell it there's error and to weight points by error (use fit report)

# First order kinetic decay function
def first_order_kinetic_decay(x, A_0, k, C):
    return A_0 * np.exp(-k * x) + C

# Load the CSV file
file_path = r'exports\CSV_exports\240226_1e-6_70W_kinetics2_consolidated_PV-Amplitudes.csv'  # Replace with your CSV file path
outputPath = r'exports\CSV_exports\221202_10A_808nm_5e-3vs0cb_consolidated_PV-Amplitudes_etc.csv'
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
fig_width = 12
fig_height = fig_width / 1.618

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, loading in enumerate(unique_loadings):
    subset = data[data['Agent Loading'] == loading]
    sns.scatterplot(x=subset['Time Value'] / 60, y=subset['Amplitude'], color=colors[i], label=f'{loading}', marker='o', s=600)

    # Add error bars
    plt.errorbar(subset['Time Value'] / 60, subset['Amplitude'], yerr=subset['Amplitude Error'], fmt='none', color=colors[i], capsize=5, elinewidth=1, capthick=2)

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
        plt.plot(x_range / 60, first_order_kinetic_decay(x_range, *best_popt), color=colors[i], linewidth=5)

plt.xlabel('time (min)', fontsize=40)
plt.ylabel('Si-H PV amplitude', fontsize=40)
plt.xticks(np.arange(0, 26, 5), fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=40)

plt.show()

# Print the fit equations and their statistics
for stat in fit_statistics:
    print(stat)

fontsize = 40

## Scatterfit for k (linear) ##
plt.figure(figsize=(10, 6))
plt.scatter(k_values.keys(), [v['k'] for v in k_values.values()], marker='o', s=100)
plt.errorbar(k_values.keys(), [v['k'] for v in k_values.values()], yerr=[v['k_error'] for v in k_values.values()], fmt='none')
fit_line_x = np.linspace(min(k_values.keys()), max(k_values.keys()), 100)
fit_line_y = np.polyval(np.polyfit(list(k_values.keys()), np.log([v['k'] for v in k_values.values()]), 1), fit_line_x)
plt.plot(fit_line_x, np.exp(fit_line_y), color='red', linestyle='--')
# plt.xlabel('agent loading (wt/wt)')
# plt.ylabel('k value')
plt.xticks(fontsize=fontsize/2)

plt.title('Scatter Plot of k Values for Each Agent Loading')
plt.grid(True)

equation = f"k = exp({np.polyfit(list(k_values.keys()), np.log([v['k'] for v in k_values.values()]), 1)[1]:.4f} * x + {np.polyfit(list(k_values.keys()), np.log([v['k'] for v in k_values.values()]), 1)[0]:.4f})"
print(f"Equation: {equation}")

# ## Scatterfit for k (log) ##
# plt.figure(figsize=(10, 6))
# plt.scatter(k_values.keys(), [v['k'] for v in k_values.values()], marker='o', s=100)
# plt.errorbar(k_values.keys(), [v['k'] for v in k_values.values()], yerr=[v['k_error'] for v in k_values.values()], fmt='none')
# plt.xscale('log')  # Setting the x-axis to logarithmic scale
# plt.xlabel('Agent Loading (wt%)')
# plt.ylabel('k Value')
# plt.title('Scatter Plot of k Values for Each Agent Loading (Logarithmic Scale)')
# plt.grid(True)


# Bar plot for k values
plt.figure(figsize=(16, 10))
x_coordinates = np.arange(len(k_values))  
plt.bar(x_coordinates, [v['k'] for v in k_values.values()], color='#295695')
plt.errorbar(x_coordinates, [v['k'] for v in k_values.values()], yerr=[v['k_error'] for v in k_values.values()], fmt='none', color='black', capsize=5)
plt.xticks(x_coordinates, k_values.keys())
plt.xlabel('agent loading (wt/wt)', fontsize=fontsize)
plt.ylabel('k value', fontsize = fontsize)
plt.xticks(fontsize=fontsize/2)
plt.yticks(fontsize=fontsize/2)
plt.legend(loc='best', fontsize=fontsize/2)
plt.grid(False)

# export the k dictionary to a csv with columns (loading, k, k_error)
k_df = pd.DataFrame.from_dict(k_values, orient='index', columns=['k', 'k_error'])
k_df['loading'] = k_df.index
k_df.to_csv(outputPath, index=False)

plt.show()