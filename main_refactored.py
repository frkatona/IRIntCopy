import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps

# CONSTANTS
csv_directory = 'CSVs'
output_directory = 'exports'
WN_NORMAL_LOW = 2957
WN_NORMAL_HIGH = 2967
WN_BASELINE_LOW = 3400
WN_BASELINE_HIGH = 3600
WN_LOW = [780, 1000, 800, 2080, 2900, 3060]
WN_HIGH = [810, 1130, 950, 2280, 2970, 3080]
GROUP_NAMES = ['Si-CH3 (symm)', 'Si-O-Si', 'Si-H (bend)', 
               'Si-H (stretch)', 'CH3', 'vinyl C-H (stretch)']

# HELPER FUNCTIONS
def WN_to_index(WN_array, WN):
    """Converts wavenumber to closest index."""
    difference_array = np.absolute(WN_array - WN)
    return difference_array.argmin()

def standardize_and_correct_baseline(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    """Converts from transmittance to absorbance and corrects the baseline."""
    if WN_raw.max() > 60:
        WN_raw /= 100
        WN_raw += 1e-9
        WN_raw = np.log10(WN_raw) * -1

    baseline_diff = WN_raw[index_baseline_low:index_baseline_high].mean()
    WN_standardized = WN_raw - baseline_diff

    WN_norm = WN_standardized[index_normal_low:index_normal_high].mean()
    WN_standardized /= WN_norm

    return WN_standardized

def integrate_peaks(WN_standardized, WN_array, WN_low, WN_high):
    """Integrates peaks from the pre-corrected absorbance array."""
    areaarray = []
    for group in range(6):
        index_low = WN_to_index(WN_array, WN_low[group])
        index_high = WN_to_index(WN_array, WN_high[group])
        area = simps(WN_standardized[index_low:index_high], WN_array[index_low:index_high])
        m = (WN_standardized[index_low] - WN_standardized[index_high]) / (WN_array[index_low] - WN_array[index_high])
        b = WN_standardized[index_low] - m * WN_array[index_low]
        baseline_y = np.array(m * WN_array[index_low:index_high] + b)
        baseline_area = simps(baseline_y, WN_array[index_low:index_high])
        areaarray.append(area - baseline_area)
    return areaarray

# MAIN SCRIPT
def main():
    filelist = sorted(glob.glob(os.path.join(csv_directory, '*.csv')))
    if not filelist:
        print("No CSV files found.")
        return

    columnname = filelist[0][0:-4]
    df_tot = pd.read_csv(filelist[0], skiprows=2, header=None, names=['cm-1', columnname])
    df_tot = df_tot.sort_values(by=['cm-1'], ignore_index=True)

    df_area = pd.DataFrame(index=GROUP_NAMES)

    WN_array = df_tot['cm-1'].to_numpy()
    index_baseline_low = WN_to_index(WN_array, WN_BASELINE_LOW)
    index_baseline_high = WN_to_index(WN_array, WN_BASELINE_HIGH)
    index_normal_low = WN_to_index(WN_array, WN_NORMAL_LOW)
    index_normal_high = WN_to_index(WN_array, WN_NORMAL_HIGH)

    for file in filelist:
        columnname = file[0:-4]
        df_add = pd.read_csv(file, skiprows=2, header=None, names=['cm-1', columnname])
        df_add = df_add.sort_values(by=['cm-1'], ignore_index=True)
        WN_raw = df_add[columnname].to_numpy()
        WN_standardized = standardize_and_correct_baseline(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high)

        df_tot[columnname] = WN_standardized
        df_area[columnname] = integrate_peaks(WN_standardized, WN_array, WN_LOW, WN_HIGH)

    # Normalize the area value for each group of each file to the area of the corresponding group from the first file
    df_areas_normalized = df_area.div(df_area.iloc[:, 0], axis=0)

    # Sort the DataFrame columns alphabetically
    df_areas_sorted = df_areas_normalized.sort_index(axis=1)

    # Save the DataFrame containing peak areas to a CSV file
    df_areas_sorted.to_csv(os.path.join(output_directory, 'normalized_peak_areas.csv'))

    # Display the non-stacked bar graph of the normalized integrated peak areas
    df_areas_sorted.transpose().plot(kind='bar', stacked=False)
    plt.title('Normalized Integrated Peak Areas')
    plt.xlabel('File')
    plt.ylabel('Normalized Area')
    plt.xticks(rotation=45)
    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(output_directory, 'normalized_peak_areas.png'))

if __name__ == "__main__":
    main()
