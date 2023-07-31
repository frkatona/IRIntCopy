from distutils.command.build_scripts import first_line_re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob, os

import numpy as np
from scipy.integrate import simps
import matplotlib.colors as mcolors

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


readpath = r'CSVs'
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))

columnname = filelist[0][0:-4]
df_tot = pd.read_csv(filelist[0], skiprows = 2, header = None, names = ['cm-1', columnname])
df_tot = df_tot.sort_values(by=['cm-1'], ignore_index = True)

WN_normal_low, WN_normal_high = 1260, 1263# 2957, 2967 CH3, goes great with Si-H
WN_baseline_low, WN_baseline_high = 3400, 3600

WN_group = 0
WN_low = [780, 800, 1000, 2080, 2900, 3060]
WN_high = [810, 950, 1130, 2280, 2970, 3080]
groupname = ['Si-CH3', 'Si-H (bend)', 'Si-O-Si', 'Si-H (stretch)', 'CH3', 'vinyl (C=C)']

df_area = pd.DataFrame(index = groupname)

WN_array = df_tot['cm-1'].to_numpy()
index_baseline_low = WN_to_index(WN_array, WN_baseline_low)
index_baseline_high = WN_to_index(WN_array, WN_baseline_high)
index_normal_low = WN_to_index(WN_array, WN_normal_low)
index_normal_high = WN_to_index(WN_array, WN_normal_high)

cmap1 = plt.get_cmap("jet")  # colormap for '0cb'
cmap2 = plt.get_cmap("jet")  # colormap for '5e-3'

for i, file in enumerate(filelist):
    columnname = file[0:-4]

    df_add = pd.read_csv(file, skiprows = 2, header = None, names = ['cm-1', columnname])
    df_add = df_add.sort_values(by=['cm-1'], ignore_index = True)
    WN_raw = df_add[columnname].to_numpy()
    WN_standardized = Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    
    df_tot[columnname] = WN_standardized
    df_area[columnname] = PeakIntegration(WN_standardized, WN_array, WN_low, WN_high)

    # Calculate the color step size
    color_step = i / len(filelist)

    if file == filelist[0]:
        if '0cb' in file:
            ax_raw = df_add.plot('cm-1', columnname, title = 'Raw Spectra', color=cmap1(color_step))
            ax_stand = df_tot.plot('cm-1', columnname, title = 'Corrected Spectra', color=cmap1(color_step))
        elif '5e-3' in file:
            ax_raw = df_add.plot('cm-1', columnname, title = 'Raw Spectra', color=cmap2(color_step))
            ax_stand = df_tot.plot('cm-1', columnname, title = 'Corrected Spectra', color=cmap2(color_step))
    else:
        if '0cb' in file:
            df_add.plot('cm-1', columnname, ax = ax_raw, color=cmap1(color_step))
            df_tot.plot('cm-1', columnname, ax = ax_stand, color=cmap1(color_step))
        elif '5e-3' in file:
            df_add.plot('cm-1', columnname, ax = ax_raw, color=cmap2(color_step))
            df_tot.plot('cm-1', columnname, ax = ax_stand, color=cmap2(color_step))

ax_area = df_area.plot.bar(title = 'Peak Areas', rot = 30)
plt.show()

