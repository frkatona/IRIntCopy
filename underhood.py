import numpy as np
from scipy.integrate import simps

def WN_to_index(WN_array, WN):
    """takes array of IR wavenumbers and translates an input WN into the index it appears at (closest to)"""
    difference_array = np.absolute(WN_array - WN)
    return difference_array.argmin()

def Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    """baseline subtraction and normalization"""
    baseline_diff = WN_raw[index_baseline_low:index_baseline_high].mean() # average across indices in np.array for baseline
    WN_standardized = WN_raw - baseline_diff

    WN_norm = WN_standardized[index_normal_low:index_normal_high].mean() # average across indices in np.array for normal
    WN_standardized /= WN_norm

    return WN_standardized

def PeakIntegration(WN_standardized, WN_array, WN_low, WN_high):
    """integrate peaks from input absorbance array (pre-corrected) according to input WN bounds (include a WN array to translate WN bounds to indices)"""
    areaarray = []
    for group in range(5):
        index_low = WN_to_index(WN_array, WN_low[group])
        index_high = WN_to_index(WN_array, WN_high[group])
        area = simps(WN_standardized[index_low:index_high], WN_array[index_low:index_high])
        m = (WN_standardized[index_low] - WN_standardized[index_high])/(WN_array[index_low] - WN_array[index_high])
        b = WN_standardized[index_low] - m * WN_array[index_low]
        baseline_y = np.array(m * WN_array[index_low:index_high] + b)
        baseline_area = simps(baseline_y, WN_array[index_low:index_high])
        areaarray.append(area - baseline_area)
    return areaarray