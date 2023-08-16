import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from main import WN_to_Index, SpectraCorrection, Extract_Filename_Metadata, Get_Convention, Get_Gradient_Color

def Consolidate_And_Plot_Spectra(readpath):
    os.chdir(readpath)
    filelist = sorted(glob.glob('*.csv'))

    # sort the file list based on the time-in-seconds metadata
    filelist.sort(key=lambda x: float(Extract_Filename_Metadata(x)[2]))

    # dataframe initialization
    columnname = filelist[0].split("_", 1)[1][:-4]
    df_tot = pd.read_csv(filelist[0], skiprows=2, header=None, names=['cm-1', columnname])
    df_tot = df_tot.sort_values(by=['cm-1'], ignore_index=True)

    # define wavenumber regions of interest (normalization, baseline correction, and peak integration)
    wn_normal_low, wn_normal_high = 1260, 1263
    wn_baseline_low, wn_baseline_high = 3400, 3600

    # convert wavenumbers to indices
    wn_array = df_tot['cm-1'].to_numpy()
    index_baseline_low = WN_to_Index(wn_array, wn_baseline_low)
    index_baseline_high = WN_to_Index(wn_array, wn_baseline_high)
    index_normal_low = WN_to_Index(wn_array, wn_normal_low)
    index_normal_high = WN_to_Index(wn_array, wn_normal_high)

    # plot initialization
    fig_raw, ax_raw = plt.subplots()
    fig_stand, ax_corrected = plt.subplots()

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

        # conditional formatting
        base_color = Get_Convention('agent-loading', agent_loading)
        color = Get_Gradient_Color(base_color, 1 - i/num_samples)  # Note the "1 -" to invert the color gradient
        df_add.plot('cm-1', columnname, ax=ax_raw, color=color)
        df_tot.plot('cm-1', columnname, ax=ax_corrected, color=color)

    # plot formatting for SPECTRA (RAW) and SPECTRA (CORRECTED)
    ax_raw.set_title('Raw Spectra')
    ax_corrected.set_title('Corrected Spectra')
    plt.show()

    # Export consolidated dataframe to CSV
    script_directory = os.path.dirname(os.path.realpath(__file__))
    exports_directory = os.path.join(script_directory, "exports")
    filename = os.path.basename(readpath) + "_consolidated.csv"
    writepath = os.path.join(exports_directory, filename)

    df_tot.to_csv(writepath, index=False)

##----------------------------MAIN CODE START----------------------------##

readpath = r"CSVs\220418_808nm_3p5A_5e-5cbPDMS"
Consolidate_And_Plot_Spectra(readpath)