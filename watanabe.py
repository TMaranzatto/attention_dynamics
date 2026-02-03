# app.py
import streamlit as st
import numpy as np
from math import pi
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from random import choice
from sympy import (
    Symbol, symbols, Matrix, im, re, exp,
    conjugate, simplify, lambdify
)
import time
st.set_page_config(page_title="Attention Dynamics", layout="wide")

st.title("N particles on the unit circle — integration of (Eq. 2.6) and the WS parameters")

# Sidebar controls
with st.sidebar:
    st.header("Simulation controls")
    N = st.number_input("Number of particles N", min_value=1, max_value=500, value=20, step=1)
    T = st.number_input("End time (seconds)", min_value=0.1, value=10.0, step=1.)
    frames = st.slider("Number of time samples", min_value=500, max_value=5000, value=1000, step=50)

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

    mult = np.matmul(V, A)
    A_eigen = np.round(np.linalg.eig(A).eigenvectors, decimals=2)
    V_eigen = np.round(np.linalg.eig(V).eigenvectors, decimals=2)
    mult_eigen = np.round(np.linalg.eig(mult).eigenvectors, decimals=2)
    st.write("V eigenvectors:", V_eigen, "A eigenvectors:", A_eigen, "A * B eigenvectors:", mult_eigen)

    st.markdown("---")

    if "seed" not in st.session_state:
        st.session_state.seed = np.random.SeedSequence().entropy
    if st.button("Randomize initial angles"):
        st.session_state.seed = np.random.SeedSequence().entropy

    st.write("Random seed (internal):", int(st.session_state.seed) % (10**9))

    w0 = V[0, 0] * A[1, 0] + V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
    w1 = V[0, 0] * A[0, 0] + V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] - V[1, 1] * A[1, 1]
    w2 =-V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
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
    c2 = - 1 / 4 * w0

    # OA ODE
    rho = symbols('ρ', real=True)
    phi = symbols('ϕ', real=True)
    R1 = im(b1 * exp(2j * phi) + b2)
    R2 = im(b3 * exp(1j * phi))
    I1 = re(b1 * exp(2j * phi) + b2)
    I2 = re(b3 * exp(1j * phi))
    I3 = re(c1 * exp(1j * phi))

    rhodot = 2 * beta * (1 - rho ** 2) * (R1 * rho + R2)
    phidot = 2 * beta * ((rho ** 2 + 1) * I1 + (rho ** 2 + 1) / rho * I2 + 2 * I3 * rho + re(c2))
    print(beta)
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

def angles_from_WS_variables(rho, phi, eta, psis):
    #in the forward map, each psi corresponds to theta mod pi
    #to start, I'll flip a coin for each psi to see if we add pi or not..
    #there has to be a more principled way to do this.
    zeta = rho * np.exp(1j * phi)
    trial = [True, False]
    def get_theta(psi):
        ret = 0.5 * np.angle((zeta + np.exp(1j * (psi + eta))) / (1. + np.conj(zeta) * np.exp(1j * (psi + eta))))
        if choice(trial):
            ret += pi
        return ret
    vectorized = np.vectorize(get_theta)
    return  vectorized(psis)

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

def random_thetas():
    # Random initial angles
    rng = np.random.default_rng(int(st.session_state.seed) % (2**32))
    thetas = rng.uniform(0, 2*np.pi, size=int(N))
    WSs = WS_variables_from_angles(thetas)
    return thetas, WSs


def random_WSs():
    rng = np.random.default_rng(int(st.session_state.seed) % (2 ** 32))
    psis = rng.uniform(0, 2*np.pi, size=int(N))
    eta_0 = 0
    phi_0 = 0
    rho_0 = .01
    thetas = angles_from_WS_variables(rho_0, phi_0, eta_0, psis)
    print(np.mean(np.exp(2j*thetas)))
    return thetas, np.r_[rho_0, phi_0, eta_0, psis]


theta0, WS_variables = random_thetas()

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

total_ws_path = sol_ws.y  # shape (N, len(t_eval))

for i in [0, 2]:
    total_ws_path[i, :] = wrap_angles(total_ws_path[i, :])
    THRESHOLD = 2 * np.pi - .1
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
ax3[0].set_ylabel("$\gamma$(t)")
ax3[0].set_ylim(0, 1)
ax3[0].grid(alpha=0.3)
ax3[0].set_xlim(0, float(T))

#this follows basin analysis from Gong-Pikovsky
a = wrap_angles(total_ws_path[1] / 2 + pi/2)
dy = np.diff(a)
mask = np.abs(dy) > np.pi - .5
full_mask = np.insert(mask, 0, False)
a_path = np.where(full_mask, np.nan, a)

b = wrap_angles(total_ws_path[1] / 2 - pi/2)
dy = np.diff(b)
mask = np.abs(dy) > np.pi - .5
full_mask = np.insert(mask, 0, False)
b_path = np.where(full_mask, np.nan, b)

ax3[1].plot(sol_ws.t, a_path, color='teal', label = r'$\Phi$/2 + $\pi/2$')
ax3[1].plot(sol_ws.t, b_path, color='cyan', label = r'$\Phi$/2 - $\pi/2$')
ax3[1].set_ylabel("Phase")
ax3[1].set_yticks([-np.pi/2, 0, np.pi/2])
ax3[1].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax3[1].set_ylim(-pi, pi)
ax3[1].grid(alpha=0.3)
ax3[1].legend()
ax3[1].set_xlim(0, float(T))

ax3[2].plot(sol_ws.t, total_ws_path[2])
ax3[2].set_ylabel("η(t)")
ax3[2].set_yticks([-np.pi / 2, 0, np.pi / 2])
ax3[2].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$"])
ax3[2].set_ylim(-pi, pi)
ax3[2].grid(alpha=0.3)
ax3[2].set_xlim(0, float(T))

ax3[2].set_xlabel("t")


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
ax.set_xlabel("t")
ax.set_ylabel("θ(t)")
ax.set_title(f"{N} sample paths of angles (0 ≤ t ≤ {T})")
ax.grid(alpha=0.3)
yticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
ax.set_yticks(yticks)
ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

st.pyplot(fig)

#
#
#now compute and plot OA vector Field animation
#
#
rhodot_f = lambdify((rho, phi), rhodot, "numpy")
phidot_f = lambdify((rho, phi), phidot, "numpy")

rho_vals = np.linspace(.05, 1, 100)
phi_vals = np.linspace(-np.pi, np.pi, 100)
RHO, PHI = np.meshgrid(rho_vals, phi_vals)

RHOdot = rhodot_f(RHO, PHI)
PHIdot = phidot_f(RHO, PHI)
if type(RHOdot) == int:
    RHOdot = RHOdot * np.ones([100,100])
if type(PHIdot) == int:
    PHIdot = PHIdot * np.ones([100,100])
speed = np.sqrt(RHOdot**2 + PHIdot**2)
lw = 5*speed / speed.max()
figOA, axOA = plt.subplots(figsize=(10, 6))
axOA.streamplot(
    RHO, PHI,
    RHOdot, PHIdot,
    density= .5,
    linewidth=.5,
    broken_streamlines=False,
    color = 'k'
)
rho_min, rho_max = 0.05, 1
axOA.set_xlabel(r"$\rho(t)$")
axOA.set_ylabel(r"$\phi(t)$")
axOA.set_xlim(rho_min, rho_max)
axOA.set_ylim(-np.pi, np.pi)
axOA.set_title("OA phase space vector field")
axOA.grid(False)
axOA.set_yticks(yticks)
axOA.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
st.pyplot(figOA)

    #REMOVE THE CODE BELOW TO STOP THE ANIMATION...
    '''
    N_OA = 400


    rho_p = np.random.uniform(rho_min, rho_max, N_OA)
    phi_p = np.random.uniform(-np.pi, np.pi, N_OA)

    dt = 0.05

    scat = axOA.scatter(rho_p, phi_p, s=10, alpha=1)

    def update(frame):
        global rho_p, phi_p

        drho = rhodot_f(rho_p, phi_p)
        dphi = phidot_f(rho_p, phi_p)

        rho_p = rho_p + dt * drho
        phi_p = wrap_angles(phi_p + dt * dphi)

        # prevent rho collapse
        rho_p = np.clip(rho_p, 0.05, 1)

        scat.set_offsets(np.column_stack((rho_p, phi_p)))
        return scat,
    print('done computing')
    ani = FuncAnimation(
        figOA,
        update,
        frames=50,
        interval=100,
        blit=False
    )
    print('done animating')
    ani.save('file_name.gif', writer=PillowWriter(fps=10))
    print('Done Converting')
    st.image('file_name.gif')
    '''
    #END ANIMATION CODE

#now plot WS
with st.expander("Show integrated WS equations"):
    st.pyplot(fig3)

#finally plot order params for completeness
with st.expander("Show order-parameters"):
    R1_t = np.mean(np.exp(1j * sol.y), axis=0)
    R2_t = np.mean(np.exp(2j * sol.y), axis=0)
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax2[0].plot(sol.t, np.abs(R1_t))
    ax2[0].set_ylabel("$|R_1(t)|$")
    ax2[0].set_ylim(0, 1)
    ax2[0].grid(alpha=0.3)
    ax2[1].plot(sol.t, np.abs(R2_t))
    ax2[1].set_ylabel("$|R_2(t)|$")
    ax2[1].set_xlabel("t")
    ax2[1].set_ylim(0, 1)
    ax2[1].grid(alpha=0.3)
    ax2[0].set_xlim(0, float(T))
    ax2[1].set_xlim(0, float(T))
    st.pyplot(fig2)
