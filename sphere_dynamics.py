import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from functools import partial
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import sys
from networkx import erdos_renyi_graph, to_numpy_array, stochastic_block_model, draw

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

def estimate_clusters(positions, distance_threshold=0.2):
    distance_matrix = pdist(positions, metric='euclidean')
    linkage_matrix = linkage(distance_matrix, method='ward')
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    cluster_sizes = defaultdict(int)
    for label in cluster_labels:
        cluster_sizes[label] += 1
    return cluster_labels, cluster_sizes

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
N = 50 # Number of oscillators
# Generate random angles for spherical coordinates
phi, theta = random_angles(N)
#set this to True if testing the stochastic block model
StochasticBM_Test = True

# Convert spherical coordinates to Cartesian coordinates
positions = spherical_to_cartesian(phi, theta)
cluster_labels, cluster_sizes = estimate_clusters(positions)
cluster_color_map = {}
colors = np.zeros((N, 3))

#set isCluster to True if testing for community clustering in Stochastic Block Model
def assign_colors(cluster_labels, cluster_sizes, isCluster = False):
    if not isCluster:
        unique_clusters = set(cluster_labels)
        sorted_clusters = sorted(unique_clusters, key=lambda c: -cluster_sizes[c])
        cmap = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))
        for i, cluster in enumerate(sorted_clusters):
            cluster_color_map[cluster] = cmap[i]
        return np.array([cluster_color_map[label] for label in cluster_labels])
    else:
        return np.array([(1,0,0,.5) for _ in range(N//2)] + [(0,1,0,.5) for _ in range(N//2)])


colors = assign_colors(cluster_labels, cluster_sizes, isCluster=StochasticBM_Test)

# Scatter plot of points on the sphere
points = ax.scatter(*positions.T, c=colors, s=50)

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
    global positions, cluster_labels, cluster_sizes, colors
    positions = spherical_to_cartesian(*random_angles(N))
    cluster_labels, cluster_sizes = estimate_clusters(positions)
    colors = assign_colors(cluster_labels, cluster_sizes, isCluster=StochasticBM_Test)
    points.set_color(colors)

restart_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
restart_button = Button(restart_ax, 'Restart')
restart_button.on_clicked(restart)

'''(Jake) This function takes as input the frame, and an arbitrary 
function that takes as input N points on the sphere and returns their
 positions (NOTE: NOT THEIR STEP dx!!).  This should 
allow quick modifications and extensions of visualizations. '''

def generic_update(frame, func):
    global positions, cluster_labels, cluster_sizes, colors
    speed = slider_speed.val
    for _ in range(int(speed)):
        positions = func(positions)
    cluster_labels, cluster_sizes = estimate_clusters(positions)
    colors = assign_colors(cluster_labels, cluster_sizes, isCluster=StochasticBM_Test)
    points.set_color(colors)
    points._offsets3d = positions.T

ax_beta = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='blue')
slider_beta = Slider(ax_beta, 'Beta', 0, 10 , valinit=1, valstep=.01)
#attention dynamics with only Q,K,V matricies  that dont change over time
def step_static(pos, Q, K, V, beta):
    dx = 0.01
    Q_pos = pos @ Q.T  # Shape: (N, 3)
    K_pos = pos @ K.T  # Shape: (N, 3)
    # Compute exponent matrix (N, N)
    exponent_matrix = beta * np.dot(Q_pos, K_pos.T)
    # Compute A and normalize each row
    Attn = np.exp(exponent_matrix)
    Attn /= Attn.sum(axis=1, keepdims=True)
    deltas = np.zeros(pos.shape)
    for i in range(len(deltas)):
        temp = dx * np.sum(Attn[i, :][:, np.newaxis] * (V @ pos.T).T, axis=0)
        deltas[i] = temp - np.dot(pos[i], temp) * pos[i]
    pos += deltas
    return pos
#the wrapper function to apply step in the animation
def static_attention_3D(frame, Q, K, V):
    beta = slider_beta.val
    assert(Q.shape == K.shape == V.shape == (3,3))
    return generic_update(frame, func=partial(step_static, Q=Q, K=K, V=V, A=A, beta=beta))

def step_feedforward(pos, Q, K, V, A, w, sigma, a, b, beta):
    #w, sigma, a, b for feed forward layer
    #w, a are dxd matrices, b is a vector in R^d
    #sigma is lipshitz function that should apply element-wise
    #A is a connection matrix for an underlying graph
    #eg. def f(x): return x+5; sigma = np.vectorize(f)
    #alternatively for a one-liner, sigma = np.vectorize(lambda x: x)

    #run (normalized) self attention dynamics iff A is identity
    SA = (np.all(A == np.identity(N)))
    dx = 0.01
    Q_pos = pos @ Q.T  # Shape: (N, 3)
    K_pos = pos @ K.T  # Shape: (N, 3)
    # Compute exponent matrix (N, N)
    exponent_matrix = (beta * np.dot(Q_pos, K_pos.T))
    # Compute A and normalize each row
    Attn = A @ np.exp(exponent_matrix).T
    if SA:
        Attn /= Attn.sum(axis=1, keepdims=True)
    else:
        Attn /= (N * math.exp(beta))
    deltas = np.zeros(pos.shape)
    feedforward = (sigma(pos @ a.T)) @ w.T + b
    for i in range(len(deltas)):
        temp = dx * (np.sum(Attn[i, :][:, np.newaxis] * pos @ V.T, axis=0) + feedforward[i])
        deltas[i] = temp - np.dot(pos[i], temp) * pos[i]
    pos += deltas
    return pos
#the wrapper function to apply step in the animation
def feedforward_attention_3D(frame, Q, K, V, A, w, sigma, a, b, SA = True):
    beta = slider_beta.val
    assert(Q.shape == K.shape == V.shape == (3,3))
    return generic_update(frame, func=partial(step_feedforward, Q=Q, K=K, V=V, A=A, w=w, sigma=sigma, a=a, b=b, beta=beta))

#value to test step function with random input matrices
def generate_cluster_plot(beta_values, step,  N_values, trials=5, T=1000):
    cluster_counts = np.zeros((len(N_values), len(beta_values)))
    Q = np.random.rand(3, 3)
    K = np.random.rand(3, 3)
    V = np.random.rand(3, 3)
    for i, n in enumerate(N_values):
        for j, beta in enumerate(beta_values):
            count = 0
            for _ in range(trials):
                pos = spherical_to_cartesian(*random_angles(n))
                for _ in range(T):
                    pos = step(pos, Q, K, V, beta)
                _, labels = estimate_clusters(pos)
                count += len(labels)
            cluster_counts[i, j] = count / trials
        print(f'Progress: {i + 1}/{len(N_values)} N values completed')
    sys.stdout.write("\n")
    fig, ax = plt.subplots()
    c = ax.imshow(cluster_counts, aspect='auto', origin='lower',
                  extent=[min(beta_values), max(beta_values), min(N_values), max(N_values)])
    ax.set_xlabel("Beta")
    ax.set_ylabel("N")
    fig.colorbar(c, ax=ax, label="Average Cluster Count")
    plt.show()

if __name__ == "__main__":
    # Commented out is the test for plotting beta vs N for a random matrix V
    #betas = [0.01*i for i in range(100)]
    #Ns = [i for i in range(5,50)]
    #generate_cluster_plot(betas, step= ..., Ns, trials=5, T=5000)
    # Below is the interactive visualization code.
    q = np.diag([1,1,1])
    k = np.diag([1,1,1])
    v = np.diag([1,1,1])
    sigma_identity = np.vectorize(lambda x: x)
    w = np.identity(3)
    a = np.identity(3)
    b = np.zeros(3)

    nodes = [25, 25]
    p = .6
    probs = [[p, 1-p], [1-p, p]]
    G = stochastic_block_model(nodes, probs)
    A = 2 * to_numpy_array(G) - np.ones((N,N))
    # Set up the animation, put your update rule as second argument
    ani = animation.FuncAnimation(fig, partial(feedforward_attention_3D, Q=q, K=k, V=v, A=A, w=w, sigma=sigma_identity, a=a, b=b), interval=50, blit=False)

    # Display the plot
    plt.show()


