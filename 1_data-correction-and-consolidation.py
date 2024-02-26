import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from main import WN_to_Index, Extract_Filename_Metadata, Get_Convention, Get_Gradient_Color
from numpy.polynomial import Polynomial

"""
step 1: 
takes raw IR data, 
applies corrections, 
consolidates to a single CSV, 
previews the changes before moving on
"""

def interpolate_to_common_wn(df, common_wn):
    """
    Interpolates the dataframe to a common set of wavenumbers (if needed)
    """
    # Interpolate
    df_interpolated = df.set_index('cm-1').reindex(common_wn).interpolate(method='linear').reset_index()
    df_interpolated.columns = ['cm-1'] + list(df.columns[1:])
    return df_interpolated

def SpectraCorrection(data):
    '''Corrects spectra for polynomial baseline drift and then normalizes'''

    # Converts to absorbance if max value suggests spectra is in transmittance
    if data.iloc[:,1].max() > 60: 
        data.iloc[:,1] = data.iloc[:,1].clip(lower=1e-8) # floor above 0 to avoid log errors
        data.iloc[:, 1] /= 100 # convert from %T to T
        data.iloc[:, 1] = np.log10(data.iloc[:, 1]) * -1 # convert from T to A

    def process_wavenumber(wavenumber, data, window=5):
        """
        Find the minimum value within `window` indices of a given wavenumber and average it
        with the four values surrounding it.
        """
        closest_index = np.abs(data['cm-1'] - wavenumber).idxmin()
        start_index = max(closest_index - window, 0)
        end_index = min(closest_index + window, len(data) - 1)
        min_index = data.iloc[:,1][start_index:end_index].idxmin()
        avg_start_index = max(min_index - 2, 0)
        avg_end_index = min(min_index + 2, len(data) - 1)
        return data.iloc[:,1][avg_start_index:avg_end_index + 1].mean()

    def CalculateBaseline(data, wavenumbers, fit_degree=3):
        """
        Apply baseline correction to the dataframe using the specified wavenumbers.
        """
        processed_values = {wn: process_wavenumber(wn, data) for wn in wavenumbers}
        p = Polynomial.fit(list(processed_values.keys()), list(processed_values.values()), fit_degree)
        # print(p)
        return p(data['cm-1'])

    def normalize_spectrum(data, corrected_absorbance, start_wn, end_wn):
        """
        Normalize the spectrum to the maximum absorbance value in the specified range.
        """
        max_absorbance = corrected_absorbance[(data['cm-1'] >= start_wn) & (data['cm-1'] <= end_wn)].max()
        return corrected_absorbance / max_absorbance

    wavenumbers = [1338, 1512, 1870, 2028, 2256, 2422, 2460, 2600, 2727, 3200, 3300, 3900, 4000]
    
    # Apply baseline correction
    baseline = CalculateBaseline(data, wavenumbers)
    corrected_absorbance = data.iloc[:,1] - baseline

    # Normalize the spectrum
    normalized_corrected_absorbance = normalize_spectrum(data, corrected_absorbance, 1255, 1270)

    return normalized_corrected_absorbance, baseline

def Consolidate_And_Plot_Spectra(readpath):
    os.chdir(readpath)
    filelist = sorted(glob.glob('*.csv'))

    # sort the file list based on the time-in-seconds metadata
    filelist.sort(key=lambda x: float(Extract_Filename_Metadata(x)[2]))

    # Define a common set of wavenumbers (this range and step is just an example, adjust accordingly)
    common_wn = np.arange(400, 4000, 1)  # from 400 to 4000 with a step of 1
    df_tot = df_baseline = pd.DataFrame({'cm-1': common_wn})

    # dataframe initialization
    columnname = os.path.splitext(filelist[0])[0]
    df_tot = pd.read_csv(filelist[0], skiprows=2, header=None, names=['cm-1', columnname])
    df_tot = df_tot.sort_values(by=['cm-1'], ignore_index=True)

    plt.rcParams.update({
        'figure.figsize': (16, 10),
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.labelsize': 40,
        'axes.titlesize': 0,  # Effectively hides titles
        # 'legend.fontsize': 0,  # Effectively hides legends
        'xtick.direction': 'out',  # Outer ticks
        'ytick.direction': 'out',  # Outer ticks
    })

    # plot initialization
    fig_raw, ax_raw = plt.subplots()
    fig_stand, ax_corrected = plt.subplots()
    fig_stand, ax_baseline = plt.subplots()

    num_samples = len(filelist)
    for i, file in enumerate(filelist):
        cure_condition, agent_identity, agent_loading, time_value, time_units = Extract_Filename_Metadata(file[:-4])
        columnname = os.path.splitext(file)[0]

        # read and correct
        df_add = pd.read_csv(file, skiprows=2, header=None, names=['cm-1', columnname])
        df_add = df_add.sort_values(by=['cm-1'], ignore_index=True)    

        wn_corrected, baseline = SpectraCorrection(df_add)

        # store corrected values in dataframe
        df_tot[columnname] = wn_corrected
        df_baseline[columnname] = baseline

        # conditional formatting
        base_color = Get_Convention('agent-loading', agent_identity + '-' + agent_loading)
        color = Get_Gradient_Color(base_color, 1 - i/num_samples)  # Note the "1 -" to invert the color gradient
        df_add.plot('cm-1', columnname, ax=ax_raw, color=color)
        df_tot.plot('cm-1', columnname, ax=ax_corrected, color=color)
        # df_tot.to_csv('test.csv', index=False)
        df_baseline.plot('cm-1', columnname, ax=ax_baseline, color=color)


    # plot formatting
    ax_raw.set_title('Raw Spectra')
    ax_corrected.set_title('Corrected Spectra')
    ax_corrected.set_xlim(800, 3300)
    ax_corrected.set_ylim(-0.25, 2)
    ax_baseline.set_title('Subtracted Baselines')

    # Export consolidated dataframe to CSV
    script_directory = os.path.dirname(os.path.realpath(__file__))
    exports_directory = os.path.join(script_directory, "exports\CSV_exports")
    filename = os.path.basename(readpath) + "_consolidated.csv"
    writepath = os.path.join(exports_directory, filename)
    df_tot.to_csv(writepath, index=False)

    plt.show()

##----------------------------MAIN CODE START----------------------------##

readpath = r"CSVs\240226_1e-6_70W_kinetics2"
Consolidate_And_Plot_Spectra(readpath)