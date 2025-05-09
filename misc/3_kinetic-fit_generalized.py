import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
step 3:
fits amplitude difference values over time to kinetic model
"""

def exp_func(t, A, k, C):
    """Exponential function used for curve fitting."""
    return A * np.exp(-k * t) + C

def Get_Convention(arg1, arg2):
    return 'blue'  # dummy color, replace with actual functionality

def Get_Gradient_Color(base_color, gradient):
    return base_color  # dummy color, replace with actual functionality

def plot_si_h_stretch_scatter_updated_with_k(readpath):
    '''Plots the Si-H (stretch) peak areas for the no-agent and CB-5e-3 samples'''
    df_peak_amplitude = pd.read_csv(readpath, index_col=0)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    """
    step 3:
    fits amplitude difference values over time to kinetic model
    """

    def exp_func(t, A, k, C):
        """Exponential function used for curve fitting."""
        return A * np.exp(-k * t) + C

    def plot_si_h_stretch_scatter_updated_with_k(readpath):
        '''Plots the Si-H (stretch) peak areas for the no-agent and CB-5e-3 samples'''
        df_peak_amplitude = pd.read_csv(readpath)

        # Extract the Si-H (stretch) peak areas and the associated times
        peak_amplitude_values = df_peak_amplitude['Amplitude'].values
        times = df_peak_amplitude['Time Value'].values

        # Extract agent loadings from column headers
        agent_loadings = df_peak_amplitude['Agent Loading'].values

        # Set the font family and size to be consistent for all plots
        plt.rcParams["font.family"] = "Segoe UI"
        plt.rcParams["font.size"] = 12

        # Constants for font sizes
        axis_label_font_size = 32
        tick_font_size = 24
        legend_font_size = 20
        font_size_equation = 20
        
        # Scatter and Fit Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Iterate over unique agent loadings
        unique_agents = np.unique(agent_loadings)
        markers = iter(['o', 'D', 's', '^', 'v'])
        colors = iter(['grey', 'red', 'blue', 'green', 'purple'])
        for agent in unique_agents:
            marker = next(markers)
            color = next(colors)
            times_agent = times[agent_loadings == agent]
            values_agent = peak_amplitude_values[agent_loadings == agent]

            # Exponential fit
            params_agent, cov_agent = curve_fit(exp_func, times_agent, values_agent, p0=[0.2, 0.01, 0.7])
            extended_time_agent = np.linspace(0, 1750, 1000)
            extended_fit_agent = exp_func(extended_time_agent, *params_agent)
        
            ax.scatter(times_agent, values_agent, color=color, edgecolors=color, linewidths=5, marker=marker, s=6*plt.rcParams['lines.markersize']**2, facecolors='none', label=agent, alpha=0.8)
            ax.plot(extended_time_agent, extended_fit_agent, color=color, linewidth=2)

            # Write the equations for the fits using the coordinates with offsets
            equation_agent = r'$A_t = {:.3g} \cdot \exp(-{:.3g} t) + {:.3g}$'.format(params_agent[0], params_agent[1], params_agent[2])    
            ax.text(times_agent[-3] + 24.95, values_agent[-3] + 0.05, equation_agent, fontsize=font_size_equation, color=color)
        
        ax.set_xlabel('time (s)', fontsize=axis_label_font_size)
        ax.set_ylabel('Si-H intensity', fontsize=axis_label_font_size)
        ax.tick_params(which='major', length=7, direction='out', labelsize=tick_font_size)
        ax.tick_params(axis='y', which='minor', length=0)
        ax.set_xticks(np.arange(0, 1751, 500))
        ax.grid(False)
        
        # Add Legend
        ax.legend(loc='upper right', fontsize=legend_font_size)
        
        # To remove the top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # To increase the thickness of the bottom and left spines
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # To adjust the thickness of the tick marks
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

        plt.tight_layout()
        plt.show()

    ##----------------------------MAIN CODE START----------------------------##

readpath = r"exports\CSV_exports\231208_4xCB-loading_KBrTransmission_ambient-cure_consolidated_PV-Amplitudes.csv"  # You need to provide the path to your CSV file    plot_si_h_stretch_scatter_updated_with_k(readpath_updated)
plot_si_h_stretch_scatter_updated_with_k(readpath)