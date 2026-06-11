import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USA model: N = 3, d = 2
# Integer candidate
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
# Simulation parameters
# ============================================================

T_total = 10000.0
T_transient = 1000.0
dt = 0.02
save_every = 5

theta0 = np.array([2.18619116, 1.68803649, 2.13395452])

# Poincare section: theta_3 = SECTION_ANGLE mod 2pi
SECTION_ANGLE = 0
CROSSING_DIRECTION = "both"   # options: "up", "down", "both"


# ============================================================
# Utilities
# ============================================================

def wrap(theta):
    """Wrap angles to [-pi, pi)."""
    return (theta + np.pi) % (2*np.pi) - np.pi


def f(theta):
    """
    USA dynamics:

        theta_dot_i =
            1/3 sum_j exp(<x_i, A x_j>) <V x_j, J x_i>.

    theta may be wrapped or unwrapped.
    """
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    dtheta = np.zeros(3)

    for i in range(3):
        xi = X[i]
        Jxi = J @ xi

        total = 0.0
        for j in range(3):
            xj = X[j]
            weight = np.exp(np.dot(xi, A @ xj))
            torque = np.dot(V @ xj, Jxi)
            total += weight * torque

        dtheta[i] = total / 3.0

    return dtheta


def rk4_step_unwrapped(theta, dt):
    """
    RK4 step without wrapping. Since f is periodic, this is safe.
    Keeping theta unwrapped makes Poincare crossing detection reliable.
    """
    k1 = f(theta)
    k2 = f(theta + 0.5*dt*k1)
    k3 = f(theta + 0.5*dt*k2)
    k4 = f(theta + dt*k3)

    return theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def crossed_section(theta_old, theta_new, section_angle=0.0, direction="both"):
    """
    Detect whether theta_3 crossed section_angle mod 2pi
    between theta_old and theta_new.

    Uses unwrapped theta_3 values.
    """
    y0 = theta_old[2] - section_angle
    y1 = theta_new[2] - section_angle

    # Which 2pi-strip are we in?
    k0 = np.floor(y0 / (2*np.pi))
    k1 = np.floor(y1 / (2*np.pi))


    if k0 == k1:
        return None

    # Determine crossing direction
    if y1 > y0:
        crossing_dir = "up"
        target_k = k0 + 1
    else:
        crossing_dir = "down"
        target_k = k0

    if direction != "both" and crossing_dir != direction:
        return None

    target = section_angle + 2*np.pi*target_k

    # Linear interpolation fraction
    alpha = (target - theta_old[2]) / (theta_new[2] - theta_old[2])

    if alpha < 0 or alpha > 1:
        return None

    theta_cross = (1 - alpha)*theta_old + alpha*theta_new
    return wrap(theta_cross)


# ============================================================
# Simulate and collect attractor + Poincare section
# ============================================================

num_steps = int(T_total / dt)
transient_steps = int(T_transient / dt)

theta = theta0.astype(float).copy()   # unwrapped state

traj = []
section = []

for step in range(num_steps):
    theta_old = theta.copy()
    theta = rk4_step_unwrapped(theta, dt)

    if step > transient_steps:
        # Save wrapped trajectory for plotting
        if step % save_every == 0:
            traj.append(wrap(theta).copy())

        # Poincare crossing detection on unwrapped coordinates
        crossing = crossed_section(
            theta_old,
            theta,
            section_angle=SECTION_ANGLE,
            direction=CROSSING_DIRECTION
        )

        if crossing is not None:
            section.append(crossing[:2])

traj = np.array(traj)
section = np.array(section)

print("Saved attractor points:", len(traj))
print("Poincare section points:", len(section))
print("Final wrapped theta:", wrap(theta))


# ============================================================
# Plot 1: 3D attractor
# ============================================================
'''
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    traj[:, 0],
    traj[:, 1],
    traj[:, 2],
    s=0.2,
    alpha=0.35
)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_zlim(-np.pi, np.pi)

ax.set_xlabel("theta_1")
ax.set_ylabel("theta_2")
ax.set_zlabel("theta_3")

ax.set_title("USA apparent attractor in angle space")
ax.view_init(elev=25, azim=35)

plt.tight_layout()
plt.show()
'''

# ============================================================
# Plot 2: Poincare section
# ============================================================

plt.figure(figsize=(7, 7))

if len(section) > 0:
    plt.scatter(
        section[:, 0],
        section[:, 1],
        s=1.2,
        alpha=0.6
    )
else:
    print("No Poincare crossings found. Try CROSSING_DIRECTION='both' or a different SECTION_ANGLE.")

plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("theta_1")
plt.ylabel("theta_2")
plt.title("Poincare section: theta_3 = 0 mod 2pi")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()