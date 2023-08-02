import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, os
import re
from scipy.integrate import simps
from numpy import log10

def WN_to_index(WN_array, WN):
    difference_array = np.absolute(WN_array - WN)
    return difference_array.argmin()

def Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    if WN_raw.max() > 60: 
        WN_raw /= 100
        WN_raw += 1e-10 
        WN_raw = np.log10(WN_raw) * -1

    baseline_diff = WN_raw[index_baseline_low:index_baseline_high].mean() 
    WN_standardized = WN_raw - baseline_diff

    WN_norm = WN_standardized[index_normal_low:index_normal_high].mean() 
    WN_standardized /= WN_norm

    return WN_standardized

def PeakIntegration(WN_standardized, WN_array, WN_low, WN_high):
    areaarray = []
    for group in range(6):
        index_low = WN_to_index(WN_array, WN_low[group])
        index_high = WN_to_index(WN_array, WN_high[group])
        area = simps(WN_standardized[index_low:index_high], WN_array[index_low:index_high])
        m = (WN_standardized[index_low] - WN_standardized[index_high])/(WN_array[index_low] - WN_array[index_high])
        b = WN_standardized[index_low] - m * WN_array[index_low]
        baseline_y = np.array(m * WN_array[index_low:index_high] + b)
        baseline_area = simps(baseline_y, WN_array[index_low:index_high])
        areaarray.append(area - baseline_area)
    return areaarray

readpath = r"CSVs\10A_5ppt-vs-0-vs-time"
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))

df_tot = pd.DataFrame()
df_tot[0] = pd.read_csv(filelist[0], skiprows=2, header=None)[0]

WN_normal_low, WN_normal_high = 1260, 1263
WN_baseline_low, WN_baseline_high = 3400, 3600

WN_group = 0
WN_low = [780, 830, 970, 2100, 2930, 3060]
WN_high = [830, 930, 1150, 2225, 3000, 3080]
groupname = ['Si-CH3', 'Si-H (bend)', 'Si-O-Si', 'Si-H (stretch)', 'CH3', 'vinyl (C=C)']

df_area = pd.DataFrame(index=groupname)

WN_array = df_tot[0].to_numpy()
index_baseline_low = WN_to_index(WN_array, WN_baseline_low)
index_baseline_high = WN_to_index(WN_array, WN_baseline_high)
index_normal_low = WN_to_index(WN_array, WN_normal_low)
index_normal_high = WN_to_index(WN_array, WN_normal_high)

num_samples = len(filelist)

fig_raw, ax_raw = plt.subplots()
fig_stand, ax_stand = plt.subplots()

# Dictionaries to store x and y values for scatter plot
scatter_data = {}

fit_equations = {}  # Store the fit equations

for i, file in enumerate(filelist):
    sample_type, time, _ = re.split('_|\.', file)
    time = int(time)  # Extracted time in seconds
    columnname = f"{sample_type}_{time}s"  # Include time in seconds in column name

    df_add = pd.read_csv(file, skiprows=2, header=None, names=[0, columnname])
    df_add = df_add.sort_values(by=[0], ignore_index=True)
    WN_raw = df_add[columnname].to_numpy()
    WN_standardized = Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    
    df_tot[columnname] = WN_standardized
    area = PeakIntegration(WN_standardized, WN_array, WN_low, WN_high)
    df_area[columnname] = area

    color = plt.cm.viridis(i/num_samples)

    df_add.plot(0, columnname, ax=ax_raw, color=color, label=columnname)
    df_tot.plot(0, columnname, ax=ax_stand, color=color, label=columnname)

    # Append scatter data to appropriate list in dictionary
    scatter_data.setdefault(sample_type, {"x": [], "y": []})
    scatter_data[sample_type]["x"].append(time)
    scatter_data[sample_type]["y"].append(area[3])  # Si-H (stretch) index is 3

ax_raw.set_title('Raw Spectra')
ax_stand.set_title('Corrected Spectra')
ax_stand.set_xlabel("cm⁻¹")
ax_stand.set_ylabel("Abs (arb.)")

fig_scatter, ax_scatter = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1, len(scatter_data)))

# Create scatter plot for each sample type
for i, (sample_type, data) in enumerate(scatter_data.items()):
    x = np.array(data["x"])
    y = np.array(data["y"])
    
    ax_scatter.scatter(x, y, color=colors[i], label=sample_type)

    # Add log fit
    log_x = log10(x[x>0])
    log_y = log10(y[x>0])
    log_fit = np.polyfit(log_x, log_y, 1)
    x_space = np.linspace(min(x), max(x), 400)
    y_space = 10**(log_fit[1]) * x_space ** log_fit[0]
    ax_scatter.plot(x_space, y_space, color=colors[i], label=f"{sample_type} log fit", linestyle='--')

    # Store the fit equation
    equation_text = f"y = {10**log_fit[1]:.4e} * x ^ {log_fit[0]:.4f}"
    fit_equations[sample_type] = equation_text

ax_scatter.set_xlabel("Time (s)")
ax_scatter.set_ylabel("Peak Area")
ax_scatter.legend()

# Print the fit equations
for sample_type, equation in fit_equations.items():
    print(f"{sample_type} fit equation: {equation}")

# Increase font size everywhere but legend
for fig in [fig_raw, fig_stand, fig_scatter]:
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(2 * item.get_fontsize())

plt.show()
