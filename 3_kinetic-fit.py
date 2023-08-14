import pandas as pd
import matplotlib.pyplot as plt
from main import Get_Convention, Get_Gradient_Color

def plot_si_h_stretch_scatter(readpath):
    # Load the integrated peak areas from the CSV
    df_area = pd.read_csv(readpath, index_col=0)

    # Extract the Si-H (stretch) peak areas and the associated times
    si_h_stretch_values = df_area.loc['Si-H (stretch)'].values
    times = [int(column.split('_')[1]) for column in df_area.columns]

    # Extract agent loadings from column headers for conditional formatting
    agent_loadings = [column.split('_')[0] for column in df_area.columns]
    unique_agent_loadings = list(set(agent_loadings))

    # Plot initialization
    fig, ax = plt.subplots()

    # Plot Si-H (stretch) peak areas for each agent loading
    for agent in unique_agent_loadings:
        mask = [agent == loading for loading in agent_loadings]
        filtered_times = [time for i, time in enumerate(times) if mask[i]]
        filtered_values = [value for i, value in enumerate(si_h_stretch_values) if mask[i]]

        # Conditional formatting
        base_color = Get_Convention('agent-loading', agent)
        colors = [Get_Gradient_Color(base_color, 1 - i/len(filtered_times)) for i in range(len(filtered_times))]
        
        ax.scatter(filtered_times, filtered_values, color=colors, label=agent)

    # Plot formatting
    ax.set_title('Si-H Stretch Peak Area over Time')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Si-H Stretch Peak Area')
    ax.legend()

    plt.show()

##----------------------------MAIN CODE START----------------------------##

readpath = r"exports\peak_areas.csv"
plot_si_h_stretch_scatter(readpath)
