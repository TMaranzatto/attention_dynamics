import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from math import log
from networkx import erdos_renyi_graph, to_numpy_array, stochastic_block_model, draw

''' (Jake) Everything until the first function is boiler plate code
for setting up the visualization.  I've made the code extendable
in the sense that any update function to the points can be defined
and used, see below.  I may package this as a class at a later time...
'''
# Set up figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

N = 50  # Number of points
phases = np.random.uniform(0, 2 * np.pi, N) # Initial random phases of points
# Plot points on a unit circle
points, = plt.plot(np.cos(phases), np.sin(phases), 'bo')
circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax.add_patch(circle)


# Add slider for animation speed
ax_speed = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
slider_speed = Slider(ax_speed, 'Speed', 1, 50, valinit=1, valstep=1)

def restart(event):
    global phases
    phases = np.random.uniform(0, 2 * np.pi, N)

# Add restart button
restart_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
restart_button = Button(restart_ax, 'Restart')
restart_button.on_clicked(restart)

'''(Jake) This function takes as input the frame, and an arbitrary 
function that takes as input N points on the circle and outputs 
their updated positions (NOTE: NOT THEIR STEP dx!!).  This should 
allow quick modifications and extensions of visualizations.'''

def generic_update(frame, func):
    """Update function for animation."""
    global phases
    speed = slider_speed.val  # Read the speed value
    for _ in range(int(speed)):
        phases = func(phases)
    points.set_data(np.cos(phases), np.sin(phases))
    return points,

# Example use case for the Kuramoto model
def kuramoto_update(frame):
    def kuramoto_step(thetas):
        # Kuramoto model parameters
        K = 0.2
        omega = 1.0  # Intrinsic frequency of oscillators
        dt = 0.05  # Time step
        # T = 200  # Total number of iterations
        """Compute the next phase for each oscillator."""
        phase_diffs = -np.subtract.outer(thetas, thetas)
        new_phases = thetas
        for i in range(len(thetas)):
            thetas[i] += dt * (omega + np.sum(np.sin(phase_diffs)[i, :]) * K / N)
        return new_phases
    return generic_update(frame, func=kuramoto_step)

ax_beta = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='blue')
slider_beta = Slider(ax_beta, 'Beta', 0, 5, valinit=1, valstep=.01)
def two_dimensional_attention(frame):
    def step(thetas):
        # Set the model parameters
        beta = slider_beta.val
        #smaller dt for more stability
        dt = 0.01
        phase_diffs = -np.subtract.outer(thetas, thetas)
        for i in range(len(thetas)):
            #using the fact that sin(y-x) = -sin(x-y)
            thetas[i] += dt * (1/N) * np.sum(np.exp(beta * np.cos(-phase_diffs)[i, :]) * np.sin(phase_diffs)[i, :])
        return thetas
    return generic_update(frame, step)

#recall that p = logn / n is the (sharp) threshold for connectivity in G(n,p)
#and connectivity is a necessary condition for clustering to a single point
p = (1.1)*log(N)/N
#A = to_numpy_array(erdos_renyi_graph(N, 1/2))
nodes = [25,25]
probs = [[0.8,0.1],[0.1,0.8]]
G = stochastic_block_model(nodes, probs)
A = to_numpy_array(G)
def random_attention(frame):
    def step(thetas):
        # Set the model parameters
        beta = slider_beta.val
        #smaller dt for more stability
        dt = 0.01
        #Set up random matrix
        phase_diffs = -np.subtract.outer(thetas, thetas)
        for i in range(len(thetas)):
            #Only sum those coordinates that are connected
            thetas[i] += dt * (1/N) * np.sum(A[i, :] * np.exp(beta * np.cos(-phase_diffs)[i, :]) * np.sin(phase_diffs)[i, :])
        return thetas
    return generic_update(frame, step)


# Create real-time animation
# Put your desired update function as the second argument
ani = animation.FuncAnimation(fig, random_attention, frames=None, interval=50, blit=True, cache_frame_data=False)
#display the animation
plt.show()

#todo: allow rendering and saving of the above as a gif


