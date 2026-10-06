"""Streamlit app for attention dynamics with time-varying A and V matrices."""

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp


st.set_page_config(page_title="Time-varying attention dynamics", layout="wide")
st.title("Attention dynamics with time-varying parameters")
st.latex(
    r"A(t)=A_1\sin(t/\epsilon)+A_2\cos(t/\epsilon),\qquad "
    r"V(t)=V_1\sin(t/\epsilon)+V_2\cos(t/\epsilon)"
)


def matrix_input(label, default, key_prefix):
    """Draw four real-valued inputs and return a 2-by-2 matrix."""
    with st.expander(f"Matrix {label} (2 x 2)", expanded=True):
        left, right = st.columns(2)
        with left:
            m00 = st.number_input(
                f"{label}[0,0]", value=float(default[0, 0]), key=f"{key_prefix}_00"
            )
            m10 = st.number_input(
                f"{label}[1,0]", value=float(default[1, 0]), key=f"{key_prefix}_10"
            )
        with right:
            m01 = st.number_input(
                f"{label}[0,1]", value=float(default[0, 1]), key=f"{key_prefix}_01"
            )
            m11 = st.number_input(
                f"{label}[1,1]", value=float(default[1, 1]), key=f"{key_prefix}_11"
            )
    return np.array([[m00, m01], [m10, m11]], dtype=float)


with st.sidebar:
    st.header("Simulation controls")
    N = int(
        st.number_input(
            "Number of particles N", min_value=1, max_value=500, value=20, step=1
        )
    )
    T = float(st.number_input("End time", min_value=0.1, value=10.0, step=1.0))
    frames = int(
        st.slider(
            "Number of time samples", min_value=500, max_value=5000, value=1000, step=50
        )
    )
    beta = float(st.slider("Inverse temperature beta", 0.0, 10.0, 1.0, 0.01))
    epsilon = float(
        st.number_input(
            "epsilon", min_value=0.001, value=1.0, step=0.05, format="%.3f"
        )
    )
    integration_steps_per_period = int(
        st.slider(
            "Solver steps per coefficient period",
            min_value=50,
            max_value=400,
            value=120,
            step=10,
            help=(
                "The w, b, and c coefficients have fastest period pi*epsilon. "
                "Larger values improve resolution but take longer to compute."
            ),
        )
    )

    st.markdown("---")
    st.subheader("Initial condition in the complex disc")
    rho_0 = float(
        st.number_input(
            "rho(0)", min_value=0.0, max_value=1.0, value=0.25, step=0.01
        )
    )
    phi_0 = float(
        st.number_input(
            "phi(0)", min_value=0.0, max_value=2 * np.pi, value=0.0, step=0.05
        )
    )

    st.markdown("---")
    identity = np.eye(2)
    zero = np.zeros((2, 2))
    if st.button("Randomize all matrix values"):
        matrix_rng = np.random.default_rng()
        for matrix_key in ("A1", "V1", "A2", "V2"):
            for entry in ("00", "01", "10", "11"):
                st.session_state[f"{matrix_key}_{entry}"] = float(
                    matrix_rng.uniform(-1.0, 1.0)
                )
    st.caption("Random matrix entries are sampled independently from [-1, 1].")
    A1 = matrix_input("A_1", identity, "A1")
    V1 = matrix_input("V_1", identity, "V1")
    A2 = matrix_input("A_2", zero, "A2")
    V2 = matrix_input("V_2", zero, "V2")

    st.markdown("---")
    if "tvp_seed" not in st.session_state:
        st.session_state.tvp_seed = int(np.random.SeedSequence().entropy) % (2**32)
    if st.button("Randomize initial particle angles"):
        st.session_state.tvp_seed = int(np.random.SeedSequence().entropy) % (2**32)
    st.caption(f"Random seed: {st.session_state.tvp_seed}")

    st.markdown("---")
    st.subheader("Save figures")
    default_figure_folder = Path(__file__).resolve().parent / "saved_figures"
    figure_folder = st.text_input(
        "Output folder", value=str(default_figure_folder), key="figure_output_folder"
    )
    save_figures = st.button("Save figures")


def time_varying_matrices(t):
    """Evaluate A(t) and V(t)."""
    phase = t / epsilon
    sine = np.sin(phase)
    cosine = np.cos(phase)
    return A1 * sine + A2 * cosine, V1 * sine + V2 * cosine


def coefficients_from_matrices(A, V):
    """Construct the w-dependent b and c coefficients for fixed A and V."""
    w0 = V[0, 0] * A[1, 0] + V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w1 = V[0, 0] * A[0, 0] + V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] - V[1, 1] * A[1, 1]
    w2 = -V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w3 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] + V[1, 1] * A[0, 1]
    w4 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] - V[1, 0] * A[0, 1] - V[1, 1] * A[0, 0]
    w5 = V[0, 0] * A[0, 0] - V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] + V[1, 1] * A[1, 1]
    w6 = V[0, 0] * A[0, 1] + V[0, 1] * A[0, 0] - V[1, 0] * A[1, 1] - V[1, 1] * A[1, 0]
    w7 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] + V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w8 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] + V[1, 0] * A[0, 1] + V[1, 1] * A[0, 0]

    b1 = (1j * w5 + w6 + w7 - 1j * w8) / 16
    b2 = (1j * w5 - w6 + w7 + 1j * w8) / 16
    b3 = (1j * w1 - w2) / 8
    c1 = (-w3 + 1j * w4) / 8
    c2 = -w0 / 4
    return b1, b2, b3, c1, c2


def time_varying_coefficients(t):
    """Evaluate the w, b, and c coefficients inherited from watanabe.py."""
    A, V = time_varying_matrices(t)
    return coefficients_from_matrices(A, V)


def homogenized_coefficients():
    """Return the exact fast-phase average of the b and c coefficients."""
    sine_pair = coefficients_from_matrices(A1, V1)
    cosine_pair = coefficients_from_matrices(A2, V2)
    return tuple(0.5 * (sine_value + cosine_value) for sine_value, cosine_value in zip(sine_pair, cosine_pair))


def oa_complex_field(z, coefficients):
    """Evaluate the complex OA field for scalar or array-valued z."""
    b1, b2, b3, c1, c2 = coefficients
    B = b1 * z + b2 * np.conj(z) + b3
    C = c1 * z + np.conj(c1) * np.conj(z) + c2
    return 2j * beta * (B * z**2 + C * z + np.conj(B))


def particle_rhs(t, thetas):
    """Original particle dynamics with coefficients evaluated at time t."""
    b1, b2, b3, c1, c2 = time_varying_coefficients(t)
    order_2 = np.mean(np.exp(2j * thetas))
    order_2_conjugate = np.conj(order_2)

    omega = np.real(beta * (c1 * order_2 + np.conj(c1) * order_2_conjugate + c2))
    forcing = 2j * beta * np.conj(
        b1 * order_2 + b2 * order_2_conjugate + b3
    )
    return np.real(omega + np.imag(forcing * np.exp(-2j * thetas)))


def disc_rhs(t, state):
    """Complex-disc OA dynamics, represented by state=(Re(z), Im(z))."""
    z = state[0] + 1j * state[1]
    z_dot = oa_complex_field(z, time_varying_coefficients(t))
    return np.array([np.real(z_dot), np.imag(z_dot)])


def wrapped_path(values):
    """Wrap angles to [0, 2*pi) and break plot lines at the branch cut."""
    result = np.mod(values, 2 * np.pi)
    jumps = np.abs(np.diff(result)) > np.pi
    result[1:][jumps] = np.nan
    return result


# Although A(t) and V(t) have period 2*pi*epsilon, the w, b, and c coefficients
# are bilinear in A and V and can therefore oscillate with period pi*epsilon.
coefficient_period = np.pi * epsilon
max_step = coefficient_period / integration_steps_per_period

# Dense output sampling prevents a well-resolved integration from looking
# aliased when the user-selected display grid is too coarse. Particle output is
# capped according to N to keep the plot responsive; the two-state disc path can
# use a larger independent grid.
plot_samples_per_period = 40
required_plot_frames = int(np.ceil(T * plot_samples_per_period / coefficient_period)) + 1
particle_frame_cap = min(50_000, max(5_000, 2_000_000 // max(N, 1)))
particle_frames = max(frames, min(required_plot_frames, particle_frame_cap))
disc_frames = max(frames, min(required_plot_frames, 50_000))
particle_t_eval = np.linspace(0.0, T, particle_frames)
disc_t_eval = np.linspace(0.0, T, disc_frames)

st.caption(
    f"Numerical resolution: maximum solver step = {max_step:.3e}; "
    f"{integration_steps_per_period} solver steps per fastest coefficient period."
)
if particle_frames > frames or disc_frames > frames:
    st.info(
        "The displayed trajectories were automatically sampled more densely "
        "than the requested minimum to resolve the epsilon-dependent oscillations."
    )
if required_plot_frames > particle_frame_cap:
    st.warning(
        "The particle display grid reached its memory-safety cap. The integration "
        "is still fully resolved, but the particle plot may visually alias at this "
        "epsilon. Reduce N or T, or increase epsilon, for a denser particle plot."
    )
if required_plot_frames > 50_000:
    st.warning(
        "The OA trajectory display grid reached 50,000 samples. The integration "
        "remains resolved, but the rendered path may omit very rapid visual detail."
    )

rng = np.random.default_rng(st.session_state.tvp_seed)
theta_0 = np.sort(rng.uniform(0.0, 2 * np.pi, size=N))
z_0 = rho_0 * np.exp(1j * phi_0)

try:
    particle_solution = solve_ivp(
        particle_rhs,
        (0.0, T),
        theta_0,
        t_eval=particle_t_eval,
        method="DOP853",
        atol=1e-8,
        rtol=1e-6,
        max_step=max_step,
    )
    disc_solution = solve_ivp(
        disc_rhs,
        (0.0, T),
        [np.real(z_0), np.imag(z_0)],
        t_eval=disc_t_eval,
        method="DOP853",
        atol=1e-9,
        rtol=1e-7,
        max_step=max_step,
    )
    if not particle_solution.success:
        raise RuntimeError(f"Particle solver failed: {particle_solution.message}")
    if not disc_solution.success:
        raise RuntimeError(f"Disc solver failed: {disc_solution.message}")
except Exception as exc:
    st.error(f"Unable to integrate the dynamics: {exc}")
    st.stop()


st.header("Individual particle trajectories")
fig_particles, ax_particles = plt.subplots(figsize=(10, 6))
particle_colors = plt.cm.hsv(np.linspace(0.0, 1.0, N, endpoint=False))
for particle_index in range(N):
    ax_particles.plot(
        particle_solution.t,
        wrapped_path(particle_solution.y[particle_index].copy()),
        color=particle_colors[particle_index],
        linewidth=1.2,
    )

ax_particles.set_xlim(0.0, T)
ax_particles.set_ylim(0.0, 2 * np.pi)
ax_particles.set_xlabel("t")
ax_particles.set_ylabel(r"$\theta_i(t)$")
ax_particles.set_title(f"{N} particle paths with time-varying A(t) and V(t)")
ax_particles.set_yticks([k * np.pi / 2 for k in range(5)])
ax_particles.set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax_particles.grid(alpha=0.3)
fig_particles.tight_layout()
st.pyplot(fig_particles)


st.header("Trajectory of the initial condition in the complex disc")
z_path = disc_solution.y[0] + 1j * disc_solution.y[1]
x_path = np.real(z_path)
y_path = np.imag(z_path)

fig_disc, ax_disc = plt.subplots(figsize=(8, 8))
boundary_angle = np.linspace(0.0, 2 * np.pi, 500)
ax_disc.plot(np.cos(boundary_angle), np.sin(boundary_angle), color="black", linewidth=1.8)
ax_disc.axhline(0.0, color="0.85", linewidth=0.8)
ax_disc.axvline(0.0, color="0.85", linewidth=0.8)

points = np.column_stack((x_path, y_path)).reshape(-1, 1, 2)
segments = np.concatenate((points[:-1], points[1:]), axis=1)
trajectory = LineCollection(segments, cmap="viridis", linewidth=2.2)
trajectory.set_array(disc_solution.t[:-1])
trajectory.set_clim(0.0, T)
ax_disc.add_collection(trajectory)
colorbar = fig_disc.colorbar(trajectory, ax=ax_disc, fraction=0.046, pad=0.04)
colorbar.set_label("t")

ax_disc.scatter(x_path[0], y_path[0], color="limegreen", edgecolor="black", s=90, zorder=3, label="start")
ax_disc.scatter(x_path[-1], y_path[-1], color="red", edgecolor="black", s=90, zorder=3, label="end")
ax_disc.set_xlim(-1.08, 1.08)
ax_disc.set_ylim(-1.08, 1.08)
ax_disc.set_aspect("equal")
ax_disc.set_xlabel(r"$\operatorname{Re} z$")
ax_disc.set_ylabel(r"$\operatorname{Im} z$")
ax_disc.set_title(r"$z(t)=\rho(t)e^{i\phi(t)}$")
ax_disc.legend(loc="upper right")
fig_disc.tight_layout()
st.pyplot(fig_disc)


st.header("Homogenized OA vector field")
st.latex(
    r"X=z\in\mathbb D,\qquad "
    r"\overline{F}(z)=\frac{1}{\pi}\int_0^\pi F(z,s)\,ds"
)
st.caption(
    "Here s = t/epsilon is the fast phase. The OA field has fast-phase period "
    "pi (physical-time period pi*epsilon) because its coefficients are bilinear "
    "in A and V."
)

averaged_coefficients = homogenized_coefficients()
field_axis = np.linspace(-0.98, 0.98, 41)
field_x, field_y = np.meshgrid(field_axis, field_axis)
field_z = field_x + 1j * field_y
field_values = oa_complex_field(field_z, averaged_coefficients)
inside_disc = field_x**2 + field_y**2 <= 0.98**2
field_u = np.where(inside_disc, np.real(field_values), np.nan)
field_v = np.where(inside_disc, np.imag(field_values), np.nan)
field_speed = np.sqrt(field_u**2 + field_v**2)

# As in watanabe.py, arrows show direction with a common visual length, while
# their colors encode the true speed of the averaged field.
normalization_floor = 1e-14
field_u_normalized = field_u / (field_speed + normalization_floor)
field_v_normalized = field_v / (field_speed + normalization_floor)
valid_arrows = (
    np.isfinite(field_u_normalized)
    & np.isfinite(field_v_normalized)
    & np.isfinite(field_speed)
)

fig_homogenized, ax_homogenized = plt.subplots(figsize=(8, 8))
speed_max = float(np.nanmax(field_speed)) if np.any(valid_arrows) else 0.0
quiver = ax_homogenized.quiver(
    field_x[valid_arrows],
    field_y[valid_arrows],
    field_u_normalized[valid_arrows],
    field_v_normalized[valid_arrows],
    field_speed[valid_arrows],
    cmap="coolwarm",
    clim=(0.0, max(speed_max, normalization_floor)),
    angles="xy",
    scale_units="xy",
    scale=18,
    width=0.004,
)
homogenized_colorbar = fig_homogenized.colorbar(
    quiver, ax=ax_homogenized, fraction=0.046, pad=0.04
)
homogenized_colorbar.set_label(r"$|\overline{F}(z)|$")
ax_homogenized.plot(
    np.cos(boundary_angle), np.sin(boundary_angle), color="black", linewidth=2
)
ax_homogenized.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
ax_homogenized.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)
ax_homogenized.set_xlim(-1.08, 1.08)
ax_homogenized.set_ylim(-1.08, 1.08)
ax_homogenized.set_aspect("equal")
ax_homogenized.set_xlabel(r"$\operatorname{Re} z$")
ax_homogenized.set_ylabel(r"$\operatorname{Im} z$")
ax_homogenized.set_title("Homogenized complex OA vector field")
fig_homogenized.tight_layout()
st.pyplot(fig_homogenized)

if save_figures:
    try:
        output_folder = Path(figure_folder).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        particle_figure_path = output_folder / f"particle_trajectores-{timestamp}.png"
        oa_figure_path = output_folder / f"OA-trajectory-{timestamp}.png"
        homogenized_figure_path = output_folder / f"homogenized-OA-field-{timestamp}.png"
        fig_particles.savefig(particle_figure_path, dpi=300, bbox_inches="tight")
        fig_disc.savefig(oa_figure_path, dpi=300, bbox_inches="tight")
        fig_homogenized.savefig(homogenized_figure_path, dpi=300, bbox_inches="tight")
        st.success(
            "Saved figures:\n\n"
            f"- {particle_figure_path}\n"
            f"- {oa_figure_path}\n"
            f"- {homogenized_figure_path}"
        )
    except Exception as exc:
        st.error(f"Unable to save the figures: {exc}")

plt.close(fig_particles)
plt.close(fig_disc)
plt.close(fig_homogenized)

final_rho = float(np.abs(z_path[-1]))
final_phi = float(np.mod(np.angle(z_path[-1]), 2 * np.pi))
st.caption(f"Final state: rho(T) = {final_rho:.6f}, phi(T) = {final_phi:.6f}")

if np.max(np.abs(z_path)) > 1.001:
    st.warning(
        "The numerical trajectory left the unit disc by more than the solver tolerance. "
        "Try increasing the number of time samples or increasing epsilon."
    )
