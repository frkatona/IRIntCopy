import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import simps

conventions = {
    'cure-condition': {
        'ambient': 'o',  # circle
        'laser': 's',  # square
        'oven': 'v'  # upside-down triangle
    },
    'agent-loading': {
        'no-agent': '#FFA500', # orange
        'AuNP': '#FF0000', # red
        ## blues for CB
        'CB-1e-3': '#281E5D', # indigo
        'CB-1e-4': '#0A1172', # navy
        'CB-1e-5': '#1338BE', # cobalt
        'CB-1e-6': '#016064', # ocean
        'CB-1e-7': '#52B2BF', # sapphire
        'CB-1e-8': '#1AA7EE', # sky
        'CB-5e-3': '#281E5D', # indigo
        'CB-5e-4': '#0A1172', # navy
        'CB-5e-5': '#1338BE', # cobalt
        'CB-5e-6': '#016064', # ocean
        'CB-5e-7': '#52B2BF', # sapphire
        'CB-5e-8': '#1AA7EE', # sky
    }
}

def Get_Gradient_Color(base_color, value):
    """
    Get a gradient color based on a base color and a value between 0 and 1.
    """
    value /= 1.5
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", [base_color, "white"], N=256)
    return cmap(value)

def Get_Convention(type, key):
    return conventions[type].get(key, None)

def WN_to_Index(wn_array, wn):
    '''Finds wavenumber index in the array'''
    difference_array = np.absolute(wn_array - wn)
    return difference_array.argmin()

def SpectraCorrection(wn_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    '''Corrects spectra for baseline drift and normalizes the data'''
    ## converts to absorbance if max value suggests spectra is in transmittance
    if wn_raw.max() > 60: 
        wn_raw /= 100
        wn_raw += 1e-10 
        wn_raw = np.log10(wn_raw) * -1

    ## baseline correction
    baseline_diff = wn_raw[index_baseline_low:index_baseline_high].mean() 
    wn_corrected = wn_raw - baseline_diff

    # normalization
    wn_norm = wn_corrected[index_normal_low:index_normal_high].mean() 
    wn_corrected /= wn_norm

    return wn_corrected

def Peak_Integration(wn_corrected, wn_array, wn_low, wn_high):
    '''Integrate peak areas using Simpson's rule approximation'''
    areas = []
    for group in range(6):
        index_low = WN_to_Index(wn_array, wn_low[group])
        index_high = WN_to_Index(wn_array, wn_high[group])
        area = simps(wn_corrected[index_low:index_high], wn_array[index_low:index_high])
        
        # Calculate baseline for the peak
        m = (wn_corrected[index_low] - wn_corrected[index_high]) / (wn_array[index_low] - wn_array[index_high])
        b = wn_corrected[index_low] - m * wn_array[index_low]
        baseline_y = np.array(m * wn_array[index_low:index_high] + b)
        baseline_area = simps(baseline_y, wn_array[index_low:index_high])
        
        areas.append(area - baseline_area)
    return areas

def Extract_Filename_Metadata(file):
    '''Extract metadata formattted as "cure-condition_agent-loading_time-in-s.csv", e.g. "laser-15W/cm2_5e-3-CB_20.csv"'''
    filename_metadata = file[:-4].split("_")
    cure_condition = filename_metadata[0]
    agent_loading = filename_metadata[1]
    time_in_seconds = filename_metadata[2]
    # time = ''.join(filter(str.isdigit, time_in_seconds))
    return cure_condition, agent_loading, time_in_seconds

##----------------------------MAIN CODE START----------------------------##
def main():

    ## read files in directory
    readpath = r"CSVs\oven_5ppt-vs-0ppt"
    os.chdir(readpath)
    filelist = sorted(glob.glob('*.csv'))

    ## dataframe initialization
    columnname = filelist[0].split("_", 1)[1][:-4]
    df_tot = pd.read_csv(filelist[0], skiprows=2, header=None, names=['cm-1', columnname])
    df_tot = df_tot.sort_values(by=['cm-1'], ignore_index=True)

    ## define wavenumber regions of interest (normalization, baseline correction, and peak integration)
    wn_normal_low, wn_normal_high = 1260, 1263
    wn_baseline_low, wn_baseline_high = 3400, 3600
    wn_low = [780, 830, 970, 2100, 2930, 3060]
    wn_high = [830, 930, 1150, 2225, 3000, 3080]
    groupname = ['Si-CH3', 'Si-H (bend)', 'Si-O-Si', 'Si-H (stretch)', 'CH3', 'vinyl (C=C)']

    df_area = pd.DataFrame(index=groupname)

    ## convert wavenumbers to indices
    wn_array = df_tot['cm-1'].to_numpy()
    index_baseline_low = WN_to_Index(wn_array, wn_baseline_low)
    index_baseline_high = WN_to_Index(wn_array, wn_baseline_high)
    index_normal_low = WN_to_Index(wn_array, wn_normal_low)
    index_normal_high = WN_to_Index(wn_array, wn_normal_high)

    ## plot initialization
    fig_raw, ax_raw = plt.subplots()
    fig_stand, ax_corrected = plt.subplots()
    color_list = []
    cubehelix_palette = plt.cm.plasma(np.linspace(0, 1, len(wn_low)))
    x_si_h_stretch_0cb, y_si_h_stretch_0cb = [], []
    x_si_h_stretch_5e3, y_si_h_stretch_5e3 = [], []

    ## processing and plotting loop
    conditional = "5e-3"
    num_samples = len(filelist)
    for i, file in enumerate(filelist):
        cure_condition, agent_loading, time_in_seconds = Extract_Filename_Metadata(file)
        columnname = file.split("_", 1)[1][:-4]
        
        # read and correct
        df_add = pd.read_csv(file, skiprows=2, header=None, names=['cm-1', columnname])
        df_add = df_add.sort_values(by=['cm-1'], ignore_index=True)
        wn_raw = df_add[columnname].to_numpy()
        wn_corrected = SpectraCorrection(wn_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)
        
        # store corrected values in dataframe
        df_tot[columnname] = wn_corrected
        area = Peak_Integration(wn_corrected, wn_array, wn_low, wn_high)
        df_area[columnname] = area

        # conditional formatting
        color = plt.cm.jet(i/num_samples) if conditional in agent_loading else plt.cm.viridis(i/num_samples)
        color_list.append(color)
        (x_si_h_stretch_5e3 if conditional in agent_loading else x_si_h_stretch_0cb).append(time_in_seconds)
        (y_si_h_stretch_5e3 if conditional in agent_loading else y_si_h_stretch_0cb).append(area[3])  # Si-H (stretch) index is 3

        # plot graphs SPECTRA (RAW) and SPECTRA (CORRECTED)
        df_add.plot('cm-1', columnname, ax=ax_raw, color=color)
        df_tot.plot('cm-1', columnname, ax=ax_corrected, color=color)

    ## add peak region cutoff lines to SPECTRA (CORRECTED)
    for j in range(len(wn_low)):
        ax_corrected.axvline(x=wn_low[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)
        ax_corrected.axvline(x=wn_high[j], color=cubehelix_palette[j], linestyle='--', linewidth=2)

    ## plot formatting for SPECTRA (RAW) and SPECTRA (CORRECTED)
    ax_raw.set_title('Raw Spectra')
    ax_corrected.set_title('Corrected Spectra')
    ax_area = df_area.plot.bar(title='Peak Areas', rot=30, color=color_list)

    ## plot formatting for Si-H (STRETCH) peak area vs time (later, export to csv and use separate plotting/fitting script for kinetics)
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(x_si_h_stretch_0cb, y_si_h_stretch_0cb, color=plt.cm.viridis(np.linspace(0, 1, len(y_si_h_stretch_0cb))))
    ax_scatter.scatter(x_si_h_stretch_5e3, y_si_h_stretch_5e3, color=plt.cm.jet(np.linspace(0, 1, len(y_si_h_stretch_5e3))))
    ax_scatter.set_title('Si-H Stretch Peak Area over Time')

    plt.show()

if __name__ == "__main__":
    main()