import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# ----------------------
# Parameters
# ----------------------
n = 50
p = 1
lambda_base = 1.0
eps = 1
source = 0

timescale = 0
# timescale < 1.0   → faster playback
# timescale > 1.0   → slower playback

np.random.seed(0)

# ----------------------
# Build random graph
# ----------------------
#G = nx.erdos_renyi_graph(n,p)

G = nx.cycle_graph(n)
for i in range(1, n):
    G.add_edge(source, i)

pos = nx.spring_layout(G)

# ----------------------
# State initialization
# ----------------------
I = np.ones(n) * lambda_base
tau = np.zeros(n)
t = 0.0

# Visualization setup
fig, ax = plt.subplots(figsize=(6,6))
norm = plt.Normalize(vmin=0, vmax=5)

def draw_graph(t, tau):
    ax.clear()
    aoi = np.minimum(t - tau, 5)  # cap AOI for color scaling

    nx.draw(
        G, pos,
        node_color=aoi,
        cmap="Reds",
        node_size=400,
        #edgelist = [],
        vmin=0, vmax=5,
        with_labels=True,
        ax=ax
    )
    ax.set_title(f"Cox Gossip — real time t = {t:.2f}")
    plt.pause(0.001)   # let matplotlib breathe

# Initial draw
draw_graph(t, tau)

print("Running real-time Cox gossip simulation... (Ctrl+C to stop)")

# ----------------------
# Real-time Gillespie simulation
# ----------------------
try:
    while True:
        total_rate = np.sum(I)
        if total_rate <= 0:
            break

        # Sample actual (physical) event time
        dt = np.random.exponential(1 / total_rate)
        t += dt

        # Real-time wait
        time.sleep(dt * timescale)

        # Choose which node fires
        v = np.random.choice(n, p=I / total_rate)

        # Reset its intensity
        I[v] = lambda_base

        # Source generates a fresh timestamp
        if v == source:
            tau[v] = t

        # Send to a random neighbor
        nbrs = list(G.neighbors(v))
        if len(nbrs) > 0:
            u = np.random.choice(nbrs)
            # Fresh update?
            if tau[v] > tau[u]:
                tau[u] = tau[v]
                I[u] += tau[v] - tau[u]


        # Update visualization
        draw_graph(t, tau)

except KeyboardInterrupt:
    print("Simulation stopped by user.")

