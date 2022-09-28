import numpy as np

def WN_to_index(WN_array, WN):
    """takes array of IR wavenumbers and translates into index values"""
    difference_array = np.absolute(WN_array - WN)
    return difference_array.argmin()

def Standardize(WN_raw, index_baseline_low, index_baseline_high, index_normal_low, index_normal_high):
    """baseline subtraction and normalization"""
    baseline_diff = WN_raw[index_baseline_low:index_baseline_high].mean() # average across indices in np.array for baseline
    WN_standardized = WN_raw - baseline_diff

    WN_norm = WN_standardized[index_normal_low:index_normal_high].mean() # average across indices in np.array for normal
    WN_standardized /= WN_norm

    return WN_standardized