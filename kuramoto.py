import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Kuramoto model parameters
N = 20  # Number of oscillators
omega = 1.0  # Intrinsic frequency of oscillators
dt = 0.05  # Time step
#T = 200  # Total number of iterations

# Initial random phases
phases = np.random.uniform(0, 2 * np.pi, N)

# Set up figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Plot oscillators on a unit circle
points, = plt.plot(np.cos(phases), np.sin(phases), 'bo', markersize=8)
circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax.add_patch(circle)

# Add slider for real-time control of K
ax_k = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
slider_k = Slider(ax_k, 'K', 0, 50, valinit=0.5, valstep=0.01)

# Add slider for animation speed
ax_speed = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
slider_speed = Slider(ax_speed, 'Speed', 1, 50, valinit=1, valstep=1)

def kuramoto_step(phases, K):
    """Compute the next phase for each oscillator."""
    global omega, dt, N
    phase_diffs = -np.subtract.outer(phases, phases)
    new_phases = phases
    for i in range(len(phases)):
        new_phases[i] += dt* (omega + np.sum(np.sin(phase_diffs)[i, :]) * K / N)
    return new_phases

def update(frame):
    """Update function for animation."""
    global phases
    K = slider_k.val  # Read the slider value
    speed = slider_speed.val  # Read the speed value
    for _ in range(int(speed)):
        phases = kuramoto_step(phases, K)
    points.set_data(np.cos(phases), np.sin(phases))
    return points,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=True)

plt.show()
