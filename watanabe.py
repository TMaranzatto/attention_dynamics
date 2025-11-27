# app.py
import streamlit as st
import numpy as np
from math import pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Attention Dynamics", layout="wide")

st.title("N particles on the unit circle — integration of (Eq. 2.6) and the WS parameters")

# Sidebar controls
with st.sidebar:
    st.header("Simulation controls")
    N = st.number_input("Number of particles N", min_value=1, max_value=500, value=20, step=1)
    T = st.number_input("End time (seconds)", min_value=0.1, value=10.0, step=1.)
    frames = st.slider("Number of time samples", min_value=50, max_value=1001, value=401, step=50)

    st.markdown("---")
    st.subheader("Inverse temperature β")
    beta = st.slider("β", min_value=0., max_value=10., value=1., step=0.01)

    st.markdown("---")
    st.subheader("Matrix V (2×2)")
    col1, col2 = st.columns(2)
    with col1:
        V11 = st.number_input("V[0,0]", value=1.0, key="V11")
        V21 = st.number_input("V[1,0]", value=0.0, key="V21")
    with col2:
        V12 = st.number_input("V[0,1]", value=0.0, key="V12")
        V22 = st.number_input("V[1,1]", value=1.0, key="V22")

    V = np.array([[V11, V12],
                  [V21, V22]], dtype=float)

    st.subheader("Matrix A (2×2)")
    col3, col4 = st.columns(2)
    with col3:
        A11 = st.number_input("A[0,0]", value=1.0, key="A11")
        A21 = st.number_input("A[1,0]", value=0.0, key="A21")
    with col4:
        A12 = st.number_input("A[0,1]", value=0.0, key="A12")
        A22 = st.number_input("A[1,1]", value=1.0, key="A22")

    A = np.array([[A11, A12],
                  [A21, A22]], dtype=float)

    st.markdown("---")
    if "seed" not in st.session_state:
        st.session_state.seed = np.random.SeedSequence().entropy
    if st.button("Randomize initial angles"):
        st.session_state.seed = np.random.SeedSequence().entropy

    st.write("Random seed (internal):", int(st.session_state.seed) % (10**9))

    w0 = V[0, 0] * A[1, 0] + V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w1 = V[0, 0] * A[0, 0] + V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] - V[1, 1] * A[1, 1]
    w2 = -V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w3 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] + V[1, 1] * A[0, 1]
    w4 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] - V[1, 0] * A[0, 1] - V[1, 1] * A[0, 0]
    w5 = V[0, 0] * A[0, 0] - V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] + V[1, 1] * A[1, 1]
    w6 = V[0, 0] * A[0, 1] + V[0, 1] * A[0, 0] - V[1, 0] * A[1, 1] - V[1, 1] * A[1, 0]
    w7 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] + V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w8 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] + V[1, 0] * A[0, 1] + V[1, 1] * A[0, 0]

    b1 = 1 / 16 * (1j * w5 + w6 + w7 - 1j * w8)
    b2 = 1 / 16 * (1j * w5 - w6 + w7 + 1j * w8)
    b3 = 1 / 8 * (1j * w1 - w2)

    c1 = 1 / 8 * (-w3 + 1j * w4)
    c2 = 1 / 4 * w0

# Helper: wrap angles to [-pi, pi]
def wrap_angles(x):
    return ((x + np.pi) % (2*np.pi)) - np.pi

def make_ws_rhs():
    """Creates the WS ensemble from the data input by user"""

    def rhs(t, ws_constants):
        #extract the data we need
        rho = ws_constants[0]
        phi = ws_constants[1]
        eta = ws_constants[2]
        psis = ws_constants[3:] #contains N constants of motion


        zeta = rho * np.exp(1j * phi)

        helper = np.vectorize(
                    lambda psi : (zeta + np.exp(1j * (psi + eta))) / (1 + np.conj(zeta) * np.exp(1j * (psi + eta)))
                    )
        R2 = np.mean(helper(psis))
        R2_conj = np.conj(R2)

        omega = beta * (c1 * R2 + np.conj(c1) * R2_conj + c2)
        H = 2j * beta * np.conj((b1 * R2 + b2 * R2_conj + b3))

        rho_dot = (1 - rho**2) * np.real(H * np.exp(-1j * phi))
        phi_dot = 2 * omega + (1 + rho**2) / rho * np.imag(H * np.exp(-1j * phi))
        eta_dot = 2 * omega + 2 * rho * np.imag(H * np.exp(-1j * phi))
        return np.r_[rho_dot, phi_dot, eta_dot, np.zeros(N)]

    return rhs

# Build rhs for ODE integration
def make_rhs():
    """
    Create a function f(t, theta) -> dtheta/dt using the user-provided expressions.
    The expressions are evaluated in a restricted environment with symbols:
      theta (numpy array), R1 (complex), R2 (complex), np, N, K1, K2
    """


    def rhs(t, thetas):
        thetas = np.asarray(thetas)
        # Compute second order parameter, in this change of variable is just average of values
        R2 = np.mean(np.exp(2j * thetas))
        R2_conj = np.conj(R2)


        omega = beta * (c1 * R2 + np.conj(c1) * R2_conj + c2)
        H = 2j * beta * np.conj((b1 * R2 + b2*R2_conj + b3))

        def fin(angle):
            return omega + np.imag(H * np.exp(-2j * angle))


        vectorized = np.vectorize(fin)

        return vectorized(thetas).astype(float)

    return rhs

def WS_variables_from_angles(thetas):
    R2 = np.mean(np.exp(2j * thetas))
    #we have many choices over these, following convention from Gong+Pikovsky 2020 https://arxiv.org/pdf/1909.07718
    eta_0 = 0
    phi_0 = np.angle(R2)
    rho_0 = np.abs(R2)
    zeta_0 = rho_0 * np.exp(1j * phi_0)
    def get_phi(theta):
        return np.angle(np.exp(-2j * eta_0) * (zeta_0 - np.exp(2j * theta)) / (np.conj(zeta_0) * np.exp(2j * theta) - 1) )
    vectorized = np.vectorize(get_phi)
    return np.r_[rho_0, phi_0, eta_0, vectorized(thetas)]

# Random initial angles
rng = np.random.default_rng(int(st.session_state.seed) % (2**32))
theta0 = rng.uniform(-np.pi/2, np.pi, size=int(N))
WS_variables = WS_variables_from_angles(theta0)

#
# Integrate ORIGINAL DYNAMICS
#

t_eval = np.linspace(0.0, float(T), int(frames))
try:
    rhs = make_rhs()
    sol = solve_ivp(rhs, (0.0, float(T)), theta0, t_eval=t_eval, method='RK45', atol=1e-8, rtol=1e-6)
    if not sol.success:
        st.error("ODE solver failed: " + str(sol.message))
except Exception as e:
    st.error(f"Error building or evaluating original RHS:\n{e}")
    st.stop()

# Wrap angles to [-pi, pi] for plotting
theta_path = wrap_angles(sol.y)  # shape (N, len(t_eval))

#
# Integrate WS DYNAMICS
#

t_eval_ws = np.linspace(0.0, float(T), int(frames))
try:
    rhs = make_ws_rhs()
    sol_ws = solve_ivp(rhs, (0.0, float(T)), WS_variables, t_eval=t_eval_ws, method='RK45', atol=1e-8, rtol=1e-6)
    if not sol_ws.success:
        st.error("ODE solver failed: " + str(sol_ws.message))
except Exception as e:
    st.error(f"Error building or evaluating WS RHS:\n{e}")
    st.stop()

total_ws_path = wrap_angles(sol_ws.y)  # shape (N, len(t_eval))

for i in [1, 2]:
    THRESHOLD = 2 * pi - 0.05
    # Calculate difference between consecutive y and x values
    param_path = total_ws_path[i, :]
    dy = np.diff(param_path)

    # Create a masked array where the condition is not met
    # The mask should align with the second point of each segment, so we need to offset it
    # We add a False at the beginning of the mask to keep the first point
    mask = np.abs(dy) > THRESHOLD
    full_mask = np.insert(mask, 0, False)  # Aligns mask with y array length
    # Rewrite the y array with NaNs where the slope is too high
    total_ws_path[i, :] = np.where(full_mask, np.nan, param_path)



# PLOT EVERYTHING
#start with WS, because we need tracers later.  This is only displayed under a clikcable bar..
fig3, ax3 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
ax3[0].plot(sol_ws.t, total_ws_path[0])
ax3[0].set_ylabel("WS amplitude ρ")
ax3[0].set_ylim(0, 1)
ax3[0].grid(alpha=0.3)

#this follows basin analysis from Gong-Pikovsky
ax3[1].plot(sol_ws.t, total_ws_path[1] / 2 + pi/2, color='teal', label = r'Φ/2 + $\pi/2$')
ax3[1].plot(sol_ws.t, total_ws_path[1] / 2 - pi/2, color='cyan', label = r'Φ/2 - $\pi/2$')
ax3[1].set_ylabel("WS phase Φ \n and tracers")
ax3[1].set_yticks([-np.pi/2, 0, np.pi/2])
ax3[1].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax3[1].set_ylim(-pi, pi)
ax3[1].grid(alpha=0.3)
ax3[1].legend()

ax3[2].plot(sol_ws.t, total_ws_path[2])
ax3[2].set_ylabel("WS parameter ρ")
ax3[2].set_yticks([-np.pi / 2, 0, np.pi / 2])
ax3[2].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax3[2].set_ylim(-pi, pi)
ax3[2].grid(alpha=0.3)

ax3[2].set_xlabel("time")


#Now the original thetas
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(theta_path.shape[0]):
    THRESHOLD = 2*pi - 0.1
    # Calculate difference between consecutive y and x values
    oscillator_path = theta_path[i, :]
    dy = np.diff(oscillator_path)

    # Create a masked array where the condition is not met
    # The mask should align with the second point of each segment, so we need to offset it
    # We add a False at the beginning of the mask to keep the first point
    mask = np.abs(dy) > THRESHOLD
    full_mask = np.insert(mask, 0, False)  # Aligns mask with y array length
    # Create a new y array with NaNs where the slope is too high
    oscillator_path = np.where(full_mask, np.nan, oscillator_path)
    ax.plot(sol.t, oscillator_path, linewidth=1)

    #plot in WS subplot as well!!!
    ax3[1].plot(sol.t, oscillator_path, linewidth=0.5, alpha=0.5, color="gray")

ax.set_xlim(0, float(T))
ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel("time")
ax.set_ylabel("angle (radians)")
ax.set_title(f"{N} sample paths of angles (0 ≤ t ≤ {T})")
ax.grid(alpha=0.3)
yticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
ax.set_yticks(yticks)
ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

st.pyplot(fig)

#now plot WS
with st.expander("Show integrated WS equations"):
    st.pyplot(fig3)

#finally plot order params for completeness
with st.expander("Show order-parameters"):
    R1_t = np.mean(np.exp(1j * sol.y), axis=0)
    R2_t = np.mean(np.exp(2j * sol.y), axis=0)
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax2[0].plot(sol.t, np.abs(R1_t))
    ax2[0].set_ylabel("|R1|(t)")
    ax2[0].set_ylim(0, 1)
    ax2[0].grid(alpha=0.3)
    ax2[1].plot(sol.t, np.abs(R2_t))
    ax2[1].set_ylabel("|R2|(t)")
    ax2[1].set_xlabel("time")
    ax2[1].set_ylim(0, 1)
    ax2[1].grid(alpha=0.3)
    st.pyplot(fig2)
