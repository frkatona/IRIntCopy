import pandas as pd
import matplotlib.pyplot as plt

# Function to plot IR spectra
def plot_ir_spectra(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path)

    # Color map for the plot lines based on "b" values
    color_map = {
        # "CB-0": "#7bf1a8",
        # "CB-1e+1": "#f58549",
        # "CB-1e-2": "#7de2d1",
        # "CB-1e-4": "#00a5cf",
        "na-0": "#006d77",
        "CB-5e-3": "#9d0208"
    }

    # Create the plot
    plt.figure(figsize=(16, 10))

    # Plot each spectrum
    for column in data.columns[1:]:
        b_value = column.split('_')[1]  # Extracting the 'b' value from the column name
        color = color_map.get(b_value, "#000000")  # Default to black if not found
        
        # Smooth the data using a rolling average with window size 5
        smoothed_data = data[column].rolling(window=5, min_periods=1).mean()
        
        # Plot the smoothed data with thicker lines
        plt.plot(data['cm-1'], smoothed_data, label=column, color=color, linewidth=1.5)

    # inset formatting
    # plt.xlim(2100, 2250)
    # plt.ylim(0, 0.035)
        
    # zoomed out formatting
    plt.xlim(850, 4000)
    plt.ylim(-0.05, 1.5)
    plt.xlabel('wavenumber (cm$^{-1}$)', fontsize=40)
    plt.ylabel('absorbance', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=20)
        
    # general formatting
    # plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
file_path = r'CSVs\221202_10A_808nm_5e-3vs0cb\test.csv'  # Replace with the path to your CSV file
plot_ir_spectra(file_path)
