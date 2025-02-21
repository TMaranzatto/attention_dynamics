import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from functools import partial

#some helper functions for projections etc.
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.vstack((phi, theta)).T

# Convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(phi, theta):
    return np.vstack((
        np.sin(theta) * np.cos(phi),  # x-coordinates
        np.sin(theta) * np.sin(phi),  # y-coordinates
        np.cos(theta)                 # z-coordinates
    )).T

def random_angles(N):
    phi = np.random.uniform(0, 2 * np.pi, N)  # Azimuthal angle
    theta = np.arccos(np.random.uniform(-1, 1, N))  # Polar angle
    return phi, theta


'''(Jake) Boiler plate code to create the figure.  Need to package this
into a class at some point...'''
# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='auto')
ax.set_box_aspect([1, 1, 1])
ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))

# Remove axis lines, labels, and grid for a clean appearance
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.xaxis.line.set_visible(False)
ax.yaxis.line.set_visible(False)
ax.zaxis.line.set_visible(False)

'''(Jake) Set N points randomly on the sphere'''
N = 20  # Number of oscillators
# Generate random angles for spherical coordinates
phi, theta = random_angles(N)

# Convert spherical coordinates to Cartesian coordinates
positions = spherical_to_cartesian(phi, theta)
# Scatter plot of points on the sphere
points = ax.scatter(*positions.T, c='b', s=50)

# Create a wireframe representation of the unit sphere
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 25))
ax.plot_wireframe(
    np.cos(u) * np.sin(v),
    np.sin(u) * np.sin(v),
    np.cos(v),
    color='gray', alpha=0.2
)

# Speed slider widget
ax_speed = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgray')
slider_speed = Slider(ax_speed, 'Speed', 1, 50, valinit=1, valstep=1)

def restart(event):
    global positions
    positions = spherical_to_cartesian(*random_angles(N))

# Add restart button
restart_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
restart_button = Button(restart_ax, 'Restart')
restart_button.on_clicked(restart)

'''(Jake) This function takes as input the frame, and an arbitrary 
function that takes as input N points on the sphere and returns their
 positions (NOTE: NOT THEIR STEP dx!!).  This should 
allow quick modifications and extensions of visualizations. '''

def generic_update(frame, func):
    global positions
    speed = slider_speed.val  # Read the speed value
    for _ in range(int(speed)):
        positions = func(positions)
    # Update plot
    points._offsets3d = positions.T

# Example function for animation with random updates
def random_update(frame):
    dx = 0.01
    def random_step(pos):
        #there has to be a better way of doing this...
        coords = np.transpose(cartesian_to_spherical(*pos.T))
        pphi = coords[0]
        ttheta = coords[1]
        # Small random perturbations in spherical coordinates
        dphi = np.random.uniform(-dx, dx, N)
        dtheta = np.random.uniform(-dx, dx, N)

        # Update angles
        pphi[:] = (pphi + dphi) % (2 * np.pi)
        ttheta[:] = np.clip(ttheta + dtheta, 0, np.pi)  # Keep within valid range

        # Convert back to Cartesian coordinates
        new_pos_x = np.sin(ttheta) * np.cos(pphi)
        new_pos_y = np.sin(ttheta) * np.sin(pphi)
        new_pos_z = np.cos(ttheta)
        return np.vstack((new_pos_x, new_pos_y, new_pos_z)).T
    return generic_update(frame, func=random_step)

ax_beta = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='blue')
slider_beta = Slider(ax_beta, 'Beta', 0, 5, valinit=1, valstep=.01)
#attention dynamics but normalized by 1/N.
def static_attention_3D(frame, Q, K, V):
    dx = 0.01
    assert(Q.shape == K.shape == V.shape == (3,3))

    def step(pos):
        beta = slider_beta.val
        Q_pos = pos @ Q.T  # Shape: (N, 3)
        K_pos = pos @ K.T  # Shape: (N, 3)
        # Compute exponent matrix (N, N)
        exponent_matrix = beta * np.dot(Q_pos, K_pos.T)
        # Compute A and normalize each row
        Attn = np.exp(exponent_matrix)
        Attn /= Attn.sum(axis=1, keepdims=True)

        deltas = np.zeros(pos.shape)
        for i in range(N):
            temp = dx * np.sum(Attn[i, :][:, np.newaxis] * (V @ pos.T).T, axis=0)
            deltas[i] = temp - np.dot(pos[i], temp)*pos[i]
        pos += deltas
        return pos
    return generic_update(frame, func=step)



# Set up the animation, put your update rule as second argument
ani = animation.FuncAnimation(fig, partial(static_attention_3D, Q = np.identity(3), K =  np.identity(3), V = np.identity(3)), frames=200, interval=50, blit=False)

# Display the plot
plt.show()
