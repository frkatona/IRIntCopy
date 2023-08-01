import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, os
from scipy.integrate import simps

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

readpath = r'CSVs/210421_AuPDMS'
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))

# Identify all unique conditions in the filelist
conditions = set(file.split("_", 1)[1][:-4] for file in filelist)
num_conditions = len(conditions)

# Map each condition to a unique color map
colormap_dict = {condition: plt.cm.get_cmap(cmap_name)
                 for condition, cmap_name in zip(conditions, plt.colormaps()[:num_conditions])}


columnname = filelist[0].split("_", 1)[1][:-4]
df_tot = pd.read_csv(filelist[0], skiprows = 2, header = None, names = ['cm-1', columnname])
df_tot = df_tot.sort_values(by=['cm-1'], ignore_index = True)

WN_normal_low, WN_normal_high = 1260, 1263
WN_baseline_low, WN_baseline_high = 3400, 3600

WN_group = 0
WN_low = [780, 830, 970, 2100, 2930, 3060]
WN_high = [830, 930, 1150, 2225, 3000, 3080]
groupname = ['Si-CH3', 'Si-H (bend)', 'Si-O-Si', 'Si-H (stretch)', 'CH3', 'vinyl (C=C)']

df_area = pd.DataFrame(index=groupname)

WN_array = df_tot['cm-1'].to_numpy()
index_baseline_low = WN_to_index(WN_array, WN_baseline_low)
index_baseline_high = WN_to_index(WN_array, WN_baseline_high)
index_normal_low = WN_to_index(WN_array, WN_normal_low)
index_normal_high = WN_to_index(WN_array, WN_normal_high)

num_samples = len(filelist)

fig_raw, ax_raw = plt.subplots()
fig_stand, ax_stand = plt.subplots()

color_list = []
cubehelix_palette = plt.cm.plasma(np.linspace(0, 1, len(WN_low)))

# Parse the condition name from the filename
condition_names = [file.split("_", 1)[0] for file in filelist]

# Define dictionaries to store colors and data for each condition
condition_colors = {condition: plt.cm.jet(i / num_samples) for i, condition in enumerate(set(condition_names))}
condition_data = {condition: {"x": [], "y": []} for condition in condition_names}

for i, file in enumerate(filelist):
    condition = file.split("_", 1)[0]
    columnname = file.split("_", 1)[1][:-4]

    df_add = pd.read_csv(file, skiprows=2, header=None, names=['cm-1', columnname])
    df_add = df_add.sort_values(by=['cm-1'], ignore_index=True)
    WN_raw = df_add[columnname].to_numpy()
    WN_standardized = Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    
    df_tot[columnname] = WN_standardized
    area = PeakIntegration(WN_standardized, WN_array, WN_low, WN_high)
    df_area[columnname] = area

    color = condition_colors[condition]
    condition_data[condition]["x"].append(i)
    condition_data[condition]["y"].append(area[3])

    color_list.append(color)
    
    df_add.plot('cm-1', columnname, ax=ax_raw, color=color)
    df_tot.plot('cm-1', columnname, ax=ax_stand, color=color)

for j in range(len(WN_low)):
    ax_stand.axvline(x=WN_low[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)
    ax_stand.axvline(x=WN_high[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)

ax_raw.set_title('Raw Spectra')
ax_stand.set_title('Corrected Spectra')

ax_area = df_area.plot.bar(title='Peak Areas', rot=30, color=color_list)

fig_scatter, ax_scatter = plt.subplots()

for condition, data in condition_data.items():
    color = condition_colors[condition]
    ax_scatter.scatter(data["x"], data["y"], color=color)

ax_scatter.set_title('Si-H Stretch Area')

# Apply logarithmic fits and plot them
for condition, data in condition_data.items():
    # Exclude zero-valued data points to avoid -inf
    x_log_fit = [x for x, y in zip(data["x"], data["y"]) if y != 0]
    y_log_fit = [y for y in data["y"] if y != 0]

    # # Fit a log curve to the data
    # try:
    #     coefficients = np.polyfit(x_log_fit, np.log10(y_log_fit), 1)
    # except np.linalg.LinAlgError:
    #     print("Error fitting the data:")
    #     print("x_log_fit:", x_log_fit)
    #     print("y_log_fit:", y_log_fit)
    #     raise

    # polynomial = np.poly1d(coefficients)
    # y_fit = 10**polynomial(x_log_fit)
    # ax_scatter.plot(x_log_fit, y_fit)

plt.show()
