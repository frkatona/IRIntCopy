import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define function for logarithmic fit
def log_func(x, a, b):
    return a * np.log(b * x)

# Data
temperature_C = [150, 125, 100, 25]  # temperature in Celsius
time_minutes = [10, 20, 45, 48*60]  # time in minutes

# Initial guess for parameters a and b
initial_guess = [50, 0.05]

# Fit the log function to the data
popt, pcov = curve_fit(log_func, temperature_C, time_minutes, p0=initial_guess)

# Generate x values for the fit line (temperature from 0 to 350 C)
temp_fit = np.linspace(0, 350, 1000)

# Calculate corresponding y values (time)
time_fit = log_func(temp_fit, *popt)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(temperature_C, time_minutes, 'bo')  # Original data
plt.plot(temp_fit, time_fit, 'r')  # Log fit

# Add labels
plt.xlabel('temperature (°c)', fontsize=20)
plt.ylabel('time (minutes)', fontsize=20)

# Display the plot
plt.grid(True)
plt.show()
