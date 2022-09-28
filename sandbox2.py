from distutils.command.build_scripts import first_line_re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import simps
import glob, os

def standardize(WN, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    """baseline subtraction and normalization"""
    baseline_diff = # average across indices in np.array for baseline
    WN_standardized -= baseline_diff

    WN_norm = # average across indices in np.array for normal
    return WN_standardized


# READ FILES #
readpath = r'CSVs\220921_apk_PDMSCompare_uncured-7A30s808nm'
os.chdir(readpath)
filelist = sorted(glob.glob('*.csv'))

# INITIALIZING #
filename = filelist[0][0:-4]
df_init = pd.read_csv(filelist[0], skiprows = 1, header = 0, names = ['cm-1', filename])
ax = df_init.plot('cm-1', filename)

df_tot = df_init['cm-1']

for file in filelist:
    filename = file[0:-4]
    df_add = standardize(pd.read_csv(file, skiprows = 1, header = 0, names = ['cm-1', filename]))
    df_add = df_add.sort_values(by=['cm-1'])
    if file != filelist[0]:
        df_add.plot('cm-1', filename, ax = ax)

    df_tot[filename] = df_add[filename]

plt.show()