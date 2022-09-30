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
filename = filelist[0][0:-4]
df_tot = pd.read_csv(filelist[0], skiprows = 2, header = 0, names = ['cm-1', filename])

for file in filelist:
    filename = file[0:-4]
    df_add = pd.read_csv(file, skiprows = 2, header = None, names = ['cm-1', filename])
    df_add = df_add.sort_values(by=['cm-1'], ignore_index = True)

    # WN_raw = df_add[filename].to_numpy()
    
    df_tot[filename] = np.linspace(1,10, 6400)
numpytest = df_add[filename].to_numpy()
df_add['new column'] = numpytest # np.linspace(1,10, 6401)
print(df_tot.head(5))
# plt.show() # enable spectra plot
# df_tot.to_csv(r'C:\Users\taekw\Desktop\1_PythonScripts\IRPeakExtract\CSVs\Output\df_tot_export.csv') # enable export