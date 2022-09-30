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

# INITIALIZE FILES AND VARIABLES #
columnname = filelist[0][0:-4]
df_tot = pd.read_csv(filelist[0], skiprows = 2, header = None, names = ['cm-1', columnname])
df_tot = df_tot.sort_values(by=['cm-1'], ignore_index = True)

# normalization and baseline correction wavenumbers
WN_normal_low = 2957
WN_normal_high = 2967
WN_baseline_low = 3400
WN_baseline_high = 3600

# functional group bounds
WN_group = 0
WN_low = [715, 940, 2290, 2900, 3060]
WN_high = [830, 1230, 2390, 2970, 3080]
groupname = ['Si-O-Si (1?)', 'Si-O-Si (2?)', 'Si-H', 'CH3', 'vinyl']
note = ['?', '?', 'more cure = lower signal', 'more cure = same signal (troubleshooting)', 'more cure = lower signal']

df_area = pd.DataFrame({'Areas': [groupname[0], groupname[1], groupname[2], groupname[3], groupname[4]]})


# IR WN -> index position conversion
WN_array = df_tot['cm-1'].to_numpy()
index_baseline_low = underhood.WN_to_index(WN_array, WN_baseline_low) # later: can do these right in the module when invokding Standardize b/c it's in same place
index_baseline_high = underhood.WN_to_index(WN_array, WN_baseline_high)
index_normal_low = underhood.WN_to_index(WN_array, WN_normal_low)
index_normal_high = underhood.WN_to_index(WN_array, WN_normal_high)

for file in filelist:
    columnname = file[0:-4]
    df_add = pd.read_csv(file, skiprows = 2, header = None, names = ['cm-1', columnname])
    df_add = df_add.sort_values(by=['cm-1'], ignore_index = True)
    WN_raw = df_add[columnname].to_numpy()
    WN_standardized = underhood.Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
    
    df_tot[columnname] = WN_standardized
    df_area[columnname] = underhood.PeakIntegration(WN_standardized, WN_array, WN_low, WN_high)

    if file == filelist[0]:
        ax_raw = df_add.plot('cm-1', columnname, title = 'Raw Spectra')
        ax_stand = df_tot.plot('cm-1', columnname, title = 'Corrected Spectra')
    else:
        df_add.plot('cm-1', columnname, ax = ax_raw)
        df_tot.plot('cm-1', columnname, ax = ax_stand)

print(df_area)
# plt.show() # enable spectra plot
# df_tot.to_csv(r'C:\Users\taekw\Desktop\1_PythonScripts\IRPeakExtract\CSVs\Output\df_tot_export.csv') # enable export