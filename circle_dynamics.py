import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

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

N = 20  # Number of points
phases = np.random.uniform(0, 2 * np.pi, N) # Initial random phases of points
# Plot points on a unit circle
points, = plt.plot(np.cos(phases), np.sin(phases), 'bo', markersize=8)
circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax.add_patch(circle)


# Add slider for animation speed
ax_speed = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
slider_speed = Slider(ax_speed, 'Speed', 1, 50, valinit=1, valstep=1)

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
            new_phases[i] += dt * (omega + np.sum(np.sin(phase_diffs)[i, :]) * K / N)
        return new_phases
    return generic_update(frame, func=kuramoto_step)

# Create real-time animation
# Put your desired update function as the second argument
ani = animation.FuncAnimation(fig, kuramoto_update, frames=None, interval=50, blit=True, cache_frame_data=False)
#display the animation
plt.show()

#todo: allow rendering and saving of the above as a gif


