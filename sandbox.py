from distutils.command.build_scripts import first_line_re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import simps
import glob, os

readpath = r'CSVs\220921_apk_PDMSCompare_uncured-7A30s808nm'
writepath = r'CSVs\Output'
plotScatter = True
plotBar = False
control_number = 5

    
# assign file paths/names
colortext = open('textfiles/colorlist_lear.txt') #alternative: colorlist.txt
output = '220513_cbPDMS_laserrots_oven.csv'
willExport = False

# change working directory to folder with CSVs of interest
os.chdir(readpath) 

# make list of names of csvs from readpath directory 
filelist = sorted(glob.glob('*.csv'))  

# normalization and baseline correction wavenumbers
WN_normal_CH_low = 2957
WN_normal_CH_high = 2967
WN_baseline_low = 3400 
WN_baseline_high = 3600

# choose wavenumber range of interest (to integrate over for bar graph)
WN_group = 0
WN_low = [715, 940, 2290, 2900, 3060]
WN_high = [830, 1230, 2390, 2970, 3080]
groupname = ['Si-O-Si (?)', 'Si-O-Si (?)', 'Si-H', 'CH3', 'vinyl']
note = ['?', '?', 'more cure = lower signal', 'more cure = same signal (troubleshooting)', 'more cure = lower signal']
    
# select type of plot to show (true = scatter, false = bar)
plotScatter = True
plotBar = False
plotBox = False
plotkind = 'line' #line, bar, barh, hist, box, scatter, etc. for plotScatter plot

# matplot formatting/scaling values
xmin = 600
xmax = 4000
width = 0.8 # primary bar plot bar width
lwidth = 2

# assign colors for differentiating overlapping spectra in scatterplots (currently cycles across custom contrast gradient)
colorlist = colortext.read().split()
colorlength = len(colorlist)

# def findskips(csv, n_checkrows):
#     """finds number of rows at start of csv that contain non-float values to skip"""
#     df_findskips = pd.read_csv(csv, names = ['wavenumber', 'val'], nrows = n_checkrows)
#     nonFloats = 0
#     for index in range(n_checkrows):
#         value = df_findskips.loc['val'][index]
#         if type(value) != 'float':
#             nonFloats += 1
#     return nonFloats


# n_skiprows = findskips(filelist[0], 10)

# print(n_skiprows)

df_findskips = pd.read_csv(filelist[0], names = ['wavenumber', 'val'], nrows = 10, skiprows = 2)

# n = 5
# index = df_findskips.iloc[n]
# print(index.dtype)

# df_findskips.sort_values(by='wavenumber')
# df_findskips.to_numpy()


if df_findskips['wavenumber'][0] < df_findskips['wavenumber'][1]:
    print('ascending')
else:
    print('descending')