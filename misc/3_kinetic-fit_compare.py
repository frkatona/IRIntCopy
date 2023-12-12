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
    '''Plots the Si-H (stretch) peak areas for the no-agent and CB-5e-3 samples'''
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

    # Set the font family and size to be consistent for all plots
    plt.rcParams["font.family"] = "Segoe UI"
    plt.rcParams["font.size"] = 12

    # Constants for font sizes
    axis_label_font_size = 32
    tick_font_size = 24
    legend_font_size = 20
    font_size_equation = 15
    
    # Scatter and Fit Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(times_0cb, values_0cb, color='grey', edgecolors='grey', linewidths=5, marker='o', s=6*plt.rcParams['lines.markersize']**2, facecolors='none', label='no-agent')
    ax.scatter(times_5e3, values_5e3, color='red', edgecolors='red', linewidths=5, marker='D', s=6*plt.rcParams['lines.markersize']**2, facecolors='none', label='CB-5e-3')
    ax.plot(extended_time_0cb, extended_fit_0cb, color='grey', linewidth=2)
    ax.plot(smooth_time_5e3, smooth_fit_5e3, color='red', linewidth=2)
    
    ax.set_xlabel('time (s)', fontsize=axis_label_font_size)
    ax.set_ylabel('Si-H intensity', fontsize=axis_label_font_size)
    ax.tick_params(which='major', length=7, direction='out', labelsize=tick_font_size)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.set_xticks(np.arange(0, 1751, 500))
    ax.grid(False)
    
    # Add Legend
    ax.legend(loc='upper right', fontsize=legend_font_size)

    ## write the equations for the fits
    # Retrieve the x and y coordinates of the third-to-last point for both sets
    x_0cb, y_0cb = times_0cb[-3], values_0cb[-3]
    x_5e3, y_5e3 = times_5e3[-3], values_5e3[-3]

    # Offsets (these values can be adjusted to position the text exactly where you want)
    offset_x = 24.95  # horizontal offset
    offset_y = 0.05  # vertical offset

    # To remove the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # To increase the thickness of the bottom and left spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # To adjust the thickness of the tick marks
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    # Write the equations for the fits using the coordinates with offsets
    equation_general = r'$A_t = A_0 \exp(-kt) + C$'
    equation_0cb = r'$A_t = {:.3g} \cdot \exp(-{:.3g} t) + {:.3g}$'.format(params_0cb[0], params_0cb[1], params_0cb[2])
    equation_5e3 = r'$A_t = {:.3g} \cdot \exp(-{:.3g} t) + {:.3g}$'.format(params_5e3[0], params_5e3[1], params_5e3[2])    
    ax.text(500, 1, equation_general, fontsize=font_size_equation*2, color='black')
    ax.text(x_0cb + offset_x, y_0cb + offset_y, equation_0cb, fontsize=font_size_equation, color='grey')
    ax.text(x_5e3 + offset_x, y_5e3 + offset_y, equation_5e3, fontsize=font_size_equation, color='red')

    plt.tight_layout()

    # Bar graph for k values
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(k_values))
    colors = ['grey', 'red']
    ax.bar(x_pos, k_values, yerr=errors, align='center', alpha=0.7, capsize=10, color=colors)
    ax.set_ylabel('k values', fontsize=axis_label_font_size)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['no-agent', 'CB-5e-3'])
    ax.grid(False)
    plt.tight_layout()
    plt.show()

##----------------------------MAIN CODE START----------------------------##

readpath_updated = r"exports\221130_cbLaserSaltPlate_8A_5e-3_consolidated_peak_areas.csv"  # You need to provide the path to your CSV file
plot_si_h_stretch_scatter_updated_with_k(readpath_updated)