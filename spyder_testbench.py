import numpy as np
import plotly.graph_objects as go
import noise

def logistic_suppression(x, upperThreshold, steepness=.01):
    return x / (1 + np.exp(steepness * (x - upperThreshold)))

np.random.seed(0)
upperThreshold = 10
lowerThreshold = 0
octaves = 4
freq = 5 * octaves
persistence = 0.5
xPoints = yPoints = 500
zoom = 100
terrain = np.zeros((xPoints, yPoints))
for _ in range(octaves):
    noise_array = np.zeros((xPoints, yPoints))
    for i in range(xPoints):
        for j in range(yPoints):
            p_temp = noise.pnoise2(i / zoom, j / zoom) * freq * persistence
            if p_temp < lowerThreshold:
                p_temp = lowerThreshold
            elif p_temp > upperThreshold:
                p_temp = logistic_suppression(p_temp, upperThreshold)
            noise_array[i, j] = p_temp
    terrain += noise_array
    freq *= 2

# Create 3D plot
fig = go.Figure(data=[go.Surface(z=terrain, colorscale='earth_r')])

# Set labels and title
fig.update_layout(
    title='Terrain',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Set z-axis limits
fig.update_layout(scene=dict(zaxis_range=[0, 200]))

# Show the plot
fig.show()