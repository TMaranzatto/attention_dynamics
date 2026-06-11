import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SAVE_ANIMATION = False
OUTPUT_FILE = "three_torus_two_trajectories.mp4"  # or ".gif"
FPS = 20
DPI = 150
# ============================================================
# Adjustable simulation parameters
# ============================================================

T = 500.0          # total simulation time
dt = 0.01          # Euler timestep
skip = 10         # plot every `skip` Euler steps
trail_len = 1500   # number of plotted history points in animation

eps = 1e-4         # size of perturbation between the two initial conditions
rng = np.random.default_rng()  # randomized each run


# ============================================================
# Candidate matrices from the Lyapunov search
# ============================================================

A = np.array([
    [0,0],
    [-2, 1]
], dtype=float)

V = np.array([
    [0,  1],
    [-2, 0]
], dtype=float)

J = np.array([
    [0.0, -1.0],
    [1.0,  0.0]
])


# ============================================================
# Vector field on T^3
# ============================================================

def wrap(theta):
    """
    Wrap angles to [-pi, pi).
    """
    return (theta + np.pi) % (2*np.pi) - np.pi


def torus_difference(theta_a, theta_b):
    """
    Minimal wrapped difference theta_a - theta_b on T^3.
    """
    return wrap(theta_a - theta_b)
def break_wrapped_segments(traj):
    """
    Insert NaNs after steps where any coordinate wraps across +/- pi.

    This prevents Matplotlib from drawing artificial long line segments
    across the plotting cube.
    """
    pieces = []

    for k in range(len(traj) - 1):
        pieces.append(traj[k])

        jump = np.abs(traj[k + 1] - traj[k])
        wrapped = np.any(jump > np.pi)

        if wrapped:
            pieces.append(np.array([np.nan, np.nan, np.nan]))

    pieces.append(traj[-1])

    return np.array(pieces)


def torus_distance(theta_a, theta_b):
    """
    Euclidean norm of the wrapped difference on T^3.
    """
    return np.linalg.norm(torus_difference(theta_a, theta_b))


def f(theta):
    """
    theta has shape (3,).
    Returns dtheta/dt for the 3-agent circle system.

    x_i = (cos theta_i, sin theta_i)
    theta_dot_i =
        1/3 sum_j <x_i, A x_j> <V x_j, J x_i>.
    """
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    dtheta = np.zeros(3)

    for i in range(3):
        xi = X[i]
        Jxi = J @ xi

        s = 0.0
        for j in range(3):
            xj = X[j]
            s += np.exp(np.dot(xi, A @ xj)) * np.dot(V @ xj, Jxi)

        dtheta[i] = s / 3.0

    return dtheta


# ============================================================
# Random nearby initial conditions
# ============================================================

theta0_a = np.random.uniform(-np.pi, np.pi, size=3)

direction = rng.normal(size=3)
direction /= np.linalg.norm(direction)

theta0_b = wrap(theta0_a + eps * direction)

print("Initial condition A:", theta0_a)
print("Initial condition B:", theta0_b)
print("Initial separation:", torus_distance(theta0_a, theta0_b))


# ============================================================
# Euler simulation
# ============================================================

num_steps = int(T / dt)
num_saved = num_steps // skip + 1

traj_a = np.zeros((num_saved, 3))
traj_b = np.zeros((num_saved, 3))
dist = np.zeros(num_saved)

theta_a = wrap(theta0_a.copy())
theta_b = wrap(theta0_b.copy())

traj_a[0] = theta_a
traj_b[0] = theta_b
dist[0] = torus_distance(theta_a, theta_b)

save_idx = 1

for k in range(1, num_steps + 1):
    theta_a = wrap(theta_a + dt * f(theta_a))
    theta_b = wrap(theta_b + dt * f(theta_b))

    if k % skip == 0:
        traj_a[save_idx] = theta_a
        traj_b[save_idx] = theta_b
        dist[save_idx] = torus_distance(theta_a, theta_b)
        save_idx += 1

times = np.linspace(0, T, num_saved)

print("Simulation complete.")
print("Saved points:", len(traj_a))
print("Final theta A:", traj_a[-1])
print("Final theta B:", traj_b[-1])
print("Final separation:", dist[-1])


# ============================================================
# 3D animation of both trajectories in angle space
# ============================================================

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_zlim(-np.pi, np.pi)

ax.set_xlabel("theta_1")
ax.set_ylabel("theta_2")
ax.set_zlabel("theta_3")

ax.set_title("Two nearby Euler trajectories on the 3-torus")

# Draw cube boundaries
for s in [-np.pi, np.pi]:
    ax.plot([-np.pi, np.pi], [-np.pi, -np.pi], [s, s], linewidth=0.5)
    ax.plot([-np.pi, np.pi], [ np.pi,  np.pi], [s, s], linewidth=0.5)
    ax.plot([-np.pi, -np.pi], [-np.pi, np.pi], [s, s], linewidth=0.5)
    ax.plot([ np.pi,  np.pi], [-np.pi, np.pi], [s, s], linewidth=0.5)

    ax.plot([-np.pi, -np.pi], [s, s], [-np.pi, np.pi], linewidth=0.5)
    ax.plot([ np.pi,  np.pi], [s, s], [-np.pi, np.pi], linewidth=0.5)
    ax.plot([s, s], [-np.pi, -np.pi], [-np.pi, np.pi], linewidth=0.5)
    ax.plot([s, s], [ np.pi,  np.pi], [-np.pi, np.pi], linewidth=0.5)

line_a, = ax.plot([], [], [], linewidth=1.2, label="trajectory A")
line_b, = ax.plot([], [], [], linewidth=1.2, label="trajectory B")

point_a, = ax.plot([], [], [], marker="o", markersize=6)
point_b, = ax.plot([], [], [], marker="o", markersize=6)

time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
dist_text = ax.text2D(0.03, 0.90, "", transform=ax.transAxes)

ax.legend(loc="upper right")


def init():
    line_a.set_data([], [])
    line_a.set_3d_properties([])

    line_b.set_data([], [])
    line_b.set_3d_properties([])

    point_a.set_data([], [])
    point_a.set_3d_properties([])

    point_b.set_data([], [])
    point_b.set_3d_properties([])

    time_text.set_text("")
    dist_text.set_text("")

    return line_a, line_b, point_a, point_b, time_text, dist_text


def update(frame):
    start = max(0, frame - trail_len)

    seg_a = traj_a[start:frame + 1]
    seg_b = traj_b[start:frame + 1]

    # Break line segments when an angle wraps across +/- pi
    seg_a_plot = break_wrapped_segments(seg_a)
    seg_b_plot = break_wrapped_segments(seg_b)

    line_a.set_data(seg_a_plot[:, 0], seg_a_plot[:, 1])
    line_a.set_3d_properties(seg_a_plot[:, 2])

    line_b.set_data(seg_b_plot[:, 0], seg_b_plot[:, 1])
    line_b.set_3d_properties(seg_b_plot[:, 2])

    point_a.set_data([traj_a[frame, 0]], [traj_a[frame, 1]])
    point_a.set_3d_properties([traj_a[frame, 2]])

    point_b.set_data([traj_b[frame, 0]], [traj_b[frame, 1]])
    point_b.set_3d_properties([traj_b[frame, 2]])

    time_text.set_text(f"t = {times[frame]:.2f}")
    dist_text.set_text(f"torus distance = {dist[frame]:.3e}")

    #ax.view_init(elev=25, azim=30 + 0.05 * frame)

    return line_a, line_b, point_a, point_b, time_text, dist_text
    start = max(0, frame - trail_len)

    seg_a = traj_a[start:frame + 1]
    seg_b = traj_b[start:frame + 1]

    line_a.set_data(seg_a[:, 0], seg_a[:, 1])
    line_a.set_3d_properties(seg_a[:, 2])

    line_b.set_data(seg_b[:, 0], seg_b[:, 1])
    line_b.set_3d_properties(seg_b[:, 2])

    point_a.set_data([traj_a[frame, 0]], [traj_a[frame, 1]])
    point_a.set_3d_properties([traj_a[frame, 2]])

    point_b.set_data([traj_b[frame, 0]], [traj_b[frame, 1]])
    point_b.set_3d_properties([traj_b[frame, 2]])

    time_text.set_text(f"t = {times[frame]:.2f}")
    dist_text.set_text(f"torus distance = {dist[frame]:.3e}")

    ax.view_init(elev=25, azim=30 + 0.05 * frame)

    return line_a, line_b, point_a, point_b, time_text, dist_text


ani = FuncAnimation(
    fig,
    update,
    frames=len(traj_a),
    init_func=init,
    interval=20,
    blit=False
)

# ============================================================
# Save and/or show animation
# ============================================================

if SAVE_ANIMATION:
    if OUTPUT_FILE.endswith(".mp4"):
        # Requires ffmpeg installed.
        # On conda: conda install -c conda-forge ffmpeg
        # On Ubuntu: sudo apt install ffmpeg
        ani.save(
            OUTPUT_FILE,
            writer="ffmpeg",
            fps=FPS,
            dpi=DPI
        )
        print(f"Saved animation to {OUTPUT_FILE}")

    elif OUTPUT_FILE.endswith(".gif"):
        # Requires pillow.
        # pip install pillow
        ani.save(
            OUTPUT_FILE,
            writer="pillow",
            fps=FPS,
            dpi=DPI
        )
        print(f"Saved animation to {OUTPUT_FILE}")

    else:
        raise ValueError("OUTPUT_FILE must end in .mp4 or .gif")

plt.show()