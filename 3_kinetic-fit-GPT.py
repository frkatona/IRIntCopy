import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_func(t, A, k, C):
    """Exponential function used for curve fitting."""
    return A * np.exp(-k * t) + C

def Get_Convention(arg1, arg2):
    return 'blue'  # dummy color, replace with actual functionality

def Get_Gradient_Color(base_color, gradient):
    return base_color  # dummy color, replace with actual functionality

def plot_si_h_stretch_scatter_updated_with_k(readpath):
    # Load the integrated peak areas from the CSV
    df_area = pd.read_csv(readpath, index_col=0)

    # Extract the Si-H (stretch) peak areas and the associated times
    si_h_stretch_values = df_area.loc['Si-H (stretch)'].values
    times = [int(column.split('_')[1]) for column in df_area.columns]

    # Extract agent loadings from column headers
    agent_loadings = [column.split('_')[0] for column in df_area.columns]

    # Filter data for the agents of interest: 'CB-5e-3' and 'no-agent'
    times_0cb = [time for i, time in enumerate(times) if agent_loadings[i] == 'no-agent']
    values_0cb = [value for i, value in enumerate(si_h_stretch_values) if agent_loadings[i] == 'no-agent']
    
    times_5e3 = [time for i, time in enumerate(times) if agent_loadings[i] == 'CB-5e-3']
    values_5e3 = [value for i, value in enumerate(si_h_stretch_values) if agent_loadings[i] == 'CB-5e-3']

    # Exponential fit
    params_0cb, cov_0cb = curve_fit(exp_func, times_0cb, values_0cb, p0=[0.2, 0.01, 0.7])
    extended_time_0cb = np.linspace(0, 1750, 1000)
    extended_fit_0cb = exp_func(extended_time_0cb, *params_0cb)
    
    params_5e3, cov_5e3 = curve_fit(exp_func, times_5e3, values_5e3, p0=[0.6, 0.01, 0.25])
    smooth_time_5e3 = np.linspace(min(times_5e3), max(times_5e3), 1000)
    smooth_fit_5e3 = exp_func(smooth_time_5e3, *params_5e3)

    # Extract the standard errors for k values (index 1 in parameters)
    errors = [np.sqrt(cov_0cb[1][1]), np.sqrt(cov_5e3[1][1])]
    k_values = [params_0cb[1], params_5e3[1]]

    # Scatter and Fit Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(times_0cb, values_0cb, color='grey', edgecolors='grey', linewidths=5, marker='o', s=6*plt.rcParams['lines.markersize']**2, facecolors='none')
    ax.scatter(times_5e3, values_5e3, color='red', edgecolors='red', linewidths=5, marker='D', s=6*plt.rcParams['lines.markersize']**2, facecolors='none')
    ax.plot(extended_time_0cb, extended_fit_0cb, color='grey', linewidth=2)
    ax.plot(smooth_time_5e3, smooth_fit_5e3, color='red', linewidth=2)
    
    ax.set_xlabel('time (s)', fontsize=32)
    ax.set_ylabel('Si-H intensity', fontsize=32)
    ax.set_xlim(0, 1750)
    ax.minorticks_on()
    ax.tick_params(which='major', length=7, direction='out')
    ax.tick_params(which='minor', length=4, direction='out')
    ax.set_xticks(np.arange(0, 1751, 500))
    ax.set_xticks(np.arange(0, 1751, 100), minor=True)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    # Bar graph for k values
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(k_values))
    colors = ['grey', 'red']
    ax.bar(x_pos, k_values, yerr=errors, align='center', alpha=0.7, capsize=10, color=colors)
    ax.set_ylabel('k values')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['no-agent', 'CB-5e-3'])
    ax.grid(False)
    plt.tight_layout()
    plt.show()

readpath_updated = r"exports\peak_areas.csv"  # You need to provide the path to your CSV file
plot_si_h_stretch_scatter_updated_with_k(readpath_updated)