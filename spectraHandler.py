import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, os
from scipy.integrate import simps
from scipy.optimize import curve_fit

def reaction_curve(t, A, k, C):
    return A * np.exp(-k * t) + C

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

readpath = r'CSVs/oven_5ppt-vs-0ppt'
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))
conditional = "5e-3"

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


# Define x arrays for each scatter plot
x_si_h_stretch_0cb = []
x_si_h_stretch_5e3 = []

y_si_h_stretch_0cb = []
y_si_h_stretch_5e3 = []

for i, file in enumerate(filelist):
    columnname = file.split("_", 1)[1][:-4]

    df_add = pd.read_csv(file, skiprows=2, header=None, names=['cm-1', columnname])
    df_add = df_add.sort_values(by=['cm-1'], ignore_index=True)
    WN_raw = df_add[columnname].to_numpy()
    WN_standardized = Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    
    df_tot[columnname] = WN_standardized
    area = PeakIntegration(WN_standardized, WN_array, WN_low, WN_high)
    df_area[columnname] = area


    if conditional in file:
        color = plt.cm.Blues(i/num_samples)
        y_si_h_stretch_5e3.append(area[3])  # Si-H (stretch) index is 3
        x_si_h_stretch_5e3.append(i)  # x values for scatter plot
    else:
        color = plt.cm.Greens(i/num_samples)
        y_si_h_stretch_0cb.append(area[3])  # Si-H (stretch) index is 3
        x_si_h_stretch_0cb.append(i)  # x values for scatter plot

    color_list.append(color)
    
    df_add.plot('cm-1', columnname, ax=ax_raw, color=color) 
    df_tot.plot('cm-1', columnname, ax=ax_stand, color=color)

# export dataframes to csv
df_area.iloc[3].to_csv('area.csv')


for j in range(len(WN_low)):
    ax_stand.axvline(x=WN_low[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)
    ax_stand.axvline(x=WN_high[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)

dark_blue = '#00008B'
grey = '#2E6930'

if conditional in file:
    color = dark_blue
    y_si_h_stretch_5e3.append(area[3])  # Si-H (stretch) index is 3
    x_si_h_stretch_5e3.append(i)  # x values for scatter plot
else:
    color = grey
    y_si_h_stretch_0cb.append(area[3])  # Si-H (stretch) index is 3
    x_si_h_stretch_0cb.append(i)  # x values for scatter plot

ax_raw.set_title('Raw Spectra')
ax_stand.set_title('Corrected Spectra')

ax_area = df_area.plot.bar(title='Peak Areas', rot=30, color=color_list)

fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(x_si_h_stretch_0cb, y_si_h_stretch_0cb, color=grey)
ax_scatter.scatter(x_si_h_stretch_5e3, y_si_h_stretch_5e3, color=dark_blue)
ax_scatter.set_title('Si-H Stretch Area')

# Apply logarithmic fits and plot them

# Exclude zero-valued data points to avoid -inf
x_si_h_stretch_0cb_log_fit = [x for x, y in zip(x_si_h_stretch_0cb, y_si_h_stretch_0cb) if y != 0]
y_si_h_stretch_0cb_log_fit = [y for y in y_si_h_stretch_0cb if y != 0]

x_si_h_stretch_5e3_log_fit = [x for x, y in zip(x_si_h_stretch_5e3, y_si_h_stretch_5e3) if y != 0]
y_si_h_stretch_5e3_log_fit = [y for y in y_si_h_stretch_5e3 if y != 0]

# Fit a log curve to the data
coefficients_0cb = np.polyfit(x_si_h_stretch_0cb_log_fit, np.log10(y_si_h_stretch_0cb_log_fit), 1)
polynomial_0cb = np.poly1d(coefficients_0cb)
y_fit_0cb = 10**polynomial_0cb(x_si_h_stretch_0cb_log_fit)
ax_scatter.plot(x_si_h_stretch_0cb_log_fit, y_fit_0cb, color=grey)

coefficients_5e3 = np.polyfit(x_si_h_stretch_5e3_log_fit, np.log10(y_si_h_stretch_5e3_log_fit), 1)
polynomial_5e3 = np.poly1d(coefficients_5e3)
y_fit_5e3 = 10**polynomial_5e3(x_si_h_stretch_5e3_log_fit)
ax_scatter.plot(x_si_h_stretch_5e3_log_fit, y_fit_5e3, color=dark_blue)

# hide legends
ax_raw.legend(loc=None)
ax_stand.legend(loc=None)
ax_area.legend(loc=None)
ax_scatter.legend(loc=None)

plt.show()