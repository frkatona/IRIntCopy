from distutils.command.build_scripts import first_line_re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import simps
import glob, os
import underhood

# READ FILES #
readpath = r'CSVs\220921_apk_PDMSCompare_uncured-7A30s808nm'
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))

# INITIALIZING #
filename = filelist[0][0:-4]
df_init = pd.read_csv(filelist[0], skiprows = 1, header = 0, names = ['cm-1', filename])
ax = df_init.plot('cm-1', filename)
df_tot = df_init['cm-1']

# normalization and baseline correction wavenumbers
WN_normal_low = 2957
WN_normal_high = 2967
WN_baseline_low = 3400 
WN_baseline_high = 3600

# functional group bounds
WN_group = 0
WN_low = [715, 940, 2290, 2900, 3060]
WN_high = [830, 1230, 2390, 2970, 3080]
groupname = ['Si-O-Si (?)', 'Si-O-Si (?)', 'Si-H', 'CH3', 'vinyl']
note = ['?', '?', 'more cure = lower signal', 'more cure = same signal (troubleshooting)', 'more cure = lower signal']

# WN -> index
WN_array = np.sort(df_tot.to_numpy())
index_baseline_low = underhood.WN_to_index(WN_array, WN_baseline_low)
index_baseline_high = underhood.WN_to_index(WN_array, WN_baseline_high)
index_normal_low = underhood.WN_to_index(WN_array, WN_normal_low)
index_normal_high = underhood.WN_to_index(WN_array, WN_normal_high)
# index_group_low = # iterate through list to make new list
# index_group_high = # iterate through list to make new list

for file in filelist:
    filename = file[0:-4]
    df_add = pd.read_csv(file, skiprows = 1, header = 0, names = ['cm-1', filename])
    df_add = df_add.sort_values(by=['cm-1'])
    WN_raw = df_add[filename].to_numpy()
    WN_standardized = underhood.Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    if file != filelist[0]:
        df_add.plot('cm-1', filename, ax = ax)
    df_tot[filename] = WN_standardized.tolist()

df_tot.plot(x = ['cm-1'], y = filename)
plt.show()