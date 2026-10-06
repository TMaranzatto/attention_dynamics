"""
Attention Dynamics on S^{d-1}
==============================

Integrates the continuous-time self-attention ODE on the unit sphere:

  Linear:   dx_k/dt = P^perp_{x_k} ( beta/n     sum_j <Ax_k, x_j> V x_j )
  Softmax:  dx_k/dt = P^perp_{x_k} ( 1/Z_k      sum_j exp(beta <Ax_k, x_j>) V x_j )

where:
  - x_k in S^{d-1}  (unit sphere)
  - P^perp_x y = y - <x, y> x  (projection onto tangent space at x)
  - Z_k = sum_j <Qx_k, Kx_j>          (linear, raw sum)
        or n                             (linear, uniform)
        or sum_j exp(beta <Qx_k, Kx_j>) (softmax, always positive)
  - Q, K, V are d x d real matrices;i A = Q^T K is the combined query-key matrx


Matrix cases (from 2D OA analysis):
  Case 1: V = I,   A random symmetric PD  -> clustering governed by definiteness of A+A^T
  Case 2: A = I,   V symmetric            -> clustering governed by top eigenvalue of V
  Case 4: A = I,   V = block-diag of [[a,b],[-b,-a]] -> Hamiltonian/bifurcation
  Random: A, V both random

Tokens are initialised uniformly on S^{d-1} via normalised Gaussian draws.
ODE solver: RK45 (default), DOP853, or Radau — selectable in sidebar.

More details about ODE solvers and stiffness:
- RK45: Explicit Runge-Kutta method, good for non-stiff problems. Fast but may struggle with stiff dynamics, leading to many function evaluations or failure.
- DOP853: Higher-order explicit method, can be more efficient for smooth problems but still struggles with stiffness.
- Radau: Implicit method designed for stiff problems. More robust when stiffness is present but computationally heavier.

Stiffness can arise if tokens rapidly cluster together, causing large gradients and requiring very small time steps for explicit solvers. If you see warnings about excessive function evaluations or suspect stiffness, try switching to Radau for more reliable integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Dynamics on Sphere Merged", layout="wide")
st.title("Attention Token Dynamics on $\\mathbb{S}^{d-1}$")
st.markdown(r"""
Integrates the continuous-time self-attention ODE on the unit sphere $\mathbb{S}^{d-1}$:

$$\dot{x}_k = P^\perp_{x_k}\!\left(\frac{1}{Z_k}\sum_{j=1}^n f\!\left(\beta\langle Ax_k, x_j\rangle\right) V x_j\right), \qquad A = Q^\top K$$

with $P^\perp_x y = y - \langle x, y\rangle x$, and $f(s) = s$ (linear) or $f(s) = e^s$ (softmax).
""")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation controls")

    n_tokens = int(st.number_input("Number of tokens  n", min_value=2, max_value=1000, value=20, step=1))
    d        = int(st.number_input("Token dimension  d", min_value=2, max_value=128, value=4, step=1))
    T        = float(st.number_input("End time  T", min_value=0.1, value=5.0, step=0.5))
    frames   = int(st.slider("Time samples", min_value=100, max_value=5000, value=500, step=50))
    compare_fig2 = st.checkbox("Enable Per-Mean Fig 1", value=False)

    st.markdown("---")
    attn_type    = st.radio("Attention type", ["Softmax  f(s) = exp(s)", "Linear  f(s) = s"])
    use_softmax  = attn_type.startswith("Softmax")
    compare_fig1 = st.checkbox("Compare Linear & Softmax in Fig 1", value=False)
    
    beta         = st.slider("β (inverse temperature)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    
    if compare_fig1:
        pass
        # st.info("Comparing both in Fig 1. The rest of the plots default to Linear.")
    elif not use_softmax:
        st.info("Linear attention: Z_k = n (uniform normalisation).")

    st.markdown("---")
    st.subheader("Matrix case")
    case = st.radio(
        "Choose case",
        ["Case 1: V=I, A arbitrary",
         "Case 2: A=I, V symmetric",
         "Case 3: A=I, V rotation",
         "Case 4: A=I, V Hamiltonian",
         "Random"]
    )
    matrix_seed = int(st.number_input("Matrix random seed", min_value=0, max_value=10**9, value=42, step=1))
    rng_mat     = np.random.default_rng(int(matrix_seed))

    st.markdown("---")

    # ── Case-specific parameters ──────────────────────────────────────────────
    if case == "Case 1: V=I, A arbitrary":
        st.markdown(r"""
**Case 1**: $V = I$, $A$ arbitrary.

The precise conditions (from OA Proposition) are:

- **Full clustering** ($\rho \to 1$): $A$ has at least one eigenvalue with
  non-negative real part. Equivalently: $\mathrm{tr}(A) > 0$, or
  $\mathrm{tr}(A) \leq 0$ and $\det(A) \leq 0$.
- **Partial sync** ($\rho \to \rho_{\mathrm{eq}} \in (0,1)$):
  $\frac{A+A^\top}{2} \prec 0$ (symmetric part of $A$ is negative definite).
        """)
        regime  = st.radio("Regime", [
            "Full clustering: tr(A) > 0",
            "Full clustering: tr(A) ≤ 0 & det(A) ≤ 0",
            "Partial sync: (A+Aᵀ)/2 ≺ 0"
        ])
        a_scale = st.slider("Off-diagonal scale", min_value=0.0, max_value=3.0, value=0.5, step=0.1,
                            help="Scale of skew-symmetric perturbation added to diagonal base.")
        Q_orth  = ortho_group.rvs(d, random_state=int(matrix_seed))
        if regime.startswith("Full clustering: tr(A) > 0"):
            eigvals_A = rng_mat.uniform(0.5, 2.0, size=d)
            A_sym     = Q_orth @ np.diag(eigvals_A) @ Q_orth.T
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A  = A_sym + skew_pert
            ev_A = np.linalg.eigvals(A)
            st.success(
                f"tr(A) = {np.trace(A):.3f} > 0, "
                f"det(A) = {np.linalg.det(A):.3f} → **full clustering**"
            )
        elif regime.startswith("Full clustering: tr(A) ≤ 0"):
            n_neg     = d - 1
            pos_val   = rng_mat.uniform(0.3, 0.8)
            neg_vals  = -rng_mat.uniform(pos_val / (n_neg - 0.5),
                                          pos_val / (n_neg - 0.5) * 1.5,
                                          size=n_neg)
            eigvals_A = np.concatenate([[pos_val], neg_vals])
            A_sym     = Q_orth @ np.diag(eigvals_A) @ Q_orth.T
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A         = A_sym + skew_pert
            tr_A  = np.trace(A)
            det_A = np.linalg.det(A)
            ev_A  = np.linalg.eigvals(A)
            has_nonneg = any(v.real >= 0 for v in ev_A)
            tr_ok  = tr_A  <= 0
            det_ok = det_A <= 0
            if tr_ok and det_ok and has_nonneg:
                st.success(
                    f"tr(A) = {tr_A:.3f} ≤ 0,  det(A) = {det_A:.3e} ≤ 0,  "
                    f"A has eigenvalue with Re ≥ 0 → **full clustering**"
                )
            else:
                st.warning(
                    f"tr(A) = {tr_A:.3f},  det(A) = {det_A:.3e} — "
                    f"conditions partially met (try adjusting seed or off-diagonal scale)"
                )
        else:
            eigvals_A = rng_mat.uniform(0.5, 2.0, size=d)
            A_sym     = Q_orth @ np.diag(-eigvals_A) @ Q_orth.T
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A  = A_sym + skew_pert
            sym_part_ev = np.linalg.eigvalsh(0.5*(A + A.T))
            sym_nd = np.all(sym_part_ev < 0)
            if sym_nd:
                sym_A = 0.5*(A + A.T)
                skw_A = 0.5*(A - A.T)
                rho_eq_approx = np.linalg.norm(skw_A, "fro") / np.linalg.norm(sym_A, "fro")
                rho_eq_approx = min(rho_eq_approx, 0.999)
                st.warning(
                    f"(A+Aᵀ)/2 ≺ 0 (all sym eigenvalues < 0) → **partial sync**, "
                    f"ρ_eq ≈ {rho_eq_approx:.3f} (2D formula generalised)"
                )
            else:
                st.error("Sym part not fully ND — increase off-diagonal scale or adjust seed")
        V = np.eye(d)
        ev_A_show = np.linalg.eigvals(A)
        ev_strs = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag)>1e-10 else f"{v.real:.3f}" for v in ev_A_show]
        st.write("A eigenvalues:", "  |  ".join(ev_strs))
        st.write(f"tr(A) = {np.trace(A):.4f},  det(A) = {np.linalg.det(A):.4f}")
        st.write("(A+Aᵀ)/2 eigenvalues:", np.round(np.linalg.eigvalsh(0.5*(A+A.T)), 3).tolist())
    
    elif case == "Case 2: A=I, V symmetric":
        st.markdown(r"""
**Case 2**: $A = I$, $V$ symmetric.

Clustering governed by the **top eigenvalue** $\lambda_1$ of $V$.
Positive $\lambda_1$ → clustering; negative → dispersion.
        """)
        top_eig     = st.slider("Top eigenvalue λ₁ of V", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        other_scale = st.slider("Other eigenvalues scale", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        Q_orth      = ortho_group.rvs(d, random_state=int(matrix_seed))
        eigvals_V = rng_mat.normal(0.0, float(other_scale), size=d)
        if d > 1:
            lam = float(top_eig)
            if lam > 0:
                cap = lam * 0.9
                eigvals_V[1:] = np.clip(eigvals_V[1:], -cap, cap)
            elif lam < 0:
                eigvals_V[1:] = np.clip(eigvals_V[1:], -abs(lam) * 2, lam - 1e-3)
            else:
                eigvals_V[1:] = -np.abs(eigvals_V[1:]) - 1e-3
        eigvals_V[0] = float(top_eig)
        A = np.eye(d)
        V = Q_orth @ np.diag(eigvals_V) @ Q_orth.T
        actual_top = np.linalg.eigvalsh(V).max()
        if float(top_eig) >= 0:
            st.success(f"Top eigenvalue of V: **{actual_top:.3f}** → clustering expected")
        else:
            st.error(f"Top eigenvalue of V: **{actual_top:.3f}** → dispersion expected")
        st.write("All V eigenvalues:", np.round(np.sort(eigvals_V)[::-1], 3).tolist())

  
    elif case == "Case 3: A=I, V rotation":
        st.markdown(r"""
**Case 3**: $A = I$, $V = aI + bJ$ where $J$ is the block-diagonal skew matrix.

In 2D this is $V = \begin{pmatrix}a & b \\ -b & a\end{pmatrix}$ with $a > 0$.
The OA dynamics simplify to:
$$\dot{\rho} = \tfrac{a\beta}{2}(1-\rho^2)\rho, \qquad \dot{\phi} = \tfrac{b\beta}{2}(\rho^2+3)$$

Since $a > 0$, $\rho \to 1$ exponentially — **always clusters** regardless of $b$.
The parameter $b$ controls **rotation speed** of the cluster phase $\phi$.
For $d > 2$: $V$ is block-diagonal with $\lfloor d/2\rfloor$ identical $2\times 2$ blocks
$\begin{pmatrix}a & b \\ -b & a\end{pmatrix}$, plus $V_{dd} = a$ if $d$ is odd.
        """)
        a_param = st.slider("a  (controls clustering rate)", min_value=0.01, max_value=3.0, value=1.0, step=0.05,
                            help="Must be > 0 for clustering. Larger a = faster convergence.")
        b_param = st.slider("b  (controls rotation)", min_value=-3.0, max_value=3.0, value=0.5, step=0.05,
                            help="b=0: no rotation. b≠0: cluster phase rotates at rate b*beta/2*(rho^2+3).")
        if a_param > 0:
            st.success(f"a = {a_param} > 0 → **clustering guaranteed**, rotation rate ∝ b = {b_param}")
        else:
            st.error("a must be > 0 for clustering guarantee")
        A = np.eye(d)
        block = np.array([[a_param, b_param], [-b_param, a_param]])
        V = np.zeros((d, d))
        for i in range(0, d - 1, 2):
            V[i:i+2, i:i+2] = block
        if d % 2 == 1:
            V[d-1, d-1] = a_param

    elif case == "Case 4: A=I, V Hamiltonian":
        st.markdown(r"""
**Case 4**: $A = I$, $V$ built from $2\times 2$ Hamiltonian blocks
$\begin{pmatrix}a & b \\ -b & -a\end{pmatrix}$.

- $a > b$: complete clustering
- $a < b$: cyclic / oscillatory (bifurcation)
- $a = b$: bifurcation point

For $d > 2$: $V$ is block-diagonal with $\lfloor d/2 \rfloor$ identical blocks.
        """)
        a_param = st.slider("a", min_value=-3.0, max_value=3.0, value=1.0, step=0.05)
        b_param = st.slider("b", min_value=-3.0, max_value=3.0, value=0.5, step=0.05)
        if a_param > b_param:
            st.success(f"a > b → **clustering regime**")
        elif a_param < b_param:
            st.warning(f"a < b → **cyclic / oscillatory regime**")
        else:
            st.error(f"a = b → **bifurcation point**")
        A     = np.eye(d)
        block = np.array([[a_param, b_param], [-b_param, -a_param]])
        V     = np.zeros((d, d))
        for i in range(0, d - 1, 2):
            V[i:i+2, i:i+2] = block
        if d % 2 == 1:
            V[d-1, d-1] = 0

    else:  # Random
        st.markdown(r"""
**Random**: $A$ and $V$ drawn from $\mathcal{N}(0, \sigma^2/d)$.

Random $A$ has no guaranteed sign structure so clustering is not guaranteed.
The app will show whether $A + A^\top$ is PD (clustering indicator for linear).
        """)
        matrix_scale = st.slider("Entry std-dev σ", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        scale = float(matrix_scale) / np.sqrt(d)
        A = rng_mat.normal(0.0, scale, (d, d))
        V = rng_mat.normal(0.0, scale, (d, d))
        ApAt     = A + A.T
        ev_ApAt  = np.linalg.eigvalsh(ApAt)
        if np.all(ev_ApAt > 0):
            st.success("A + Aᵀ ≻ 0 → clustering expected (linear)")
        elif np.all(ev_ApAt < 0):
            st.error("A + Aᵀ ≺ 0 → dispersion expected (linear)")
        else:
            st.warning("A + Aᵀ indefinite → mixed behavior (linear)")

    st.markdown("---")
    st.subheader("Token initialisation")
    if "token_seed" not in st.session_state:
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    if st.button("Randomise initial tokens"):
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    st.write("Token seed:", st.session_state.token_seed)

    st.markdown("---")
    solver_choice = st.selectbox(
        "ODE solver", ["RK45", "DOP853", "Radau"], index=0,
        help="RK45/DOP853: explicit, fast. Radau: implicit, use if stiffness warnings appear."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Show matrices
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Show A and V matrices and eigenvalues"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**A** ({d}×{d})")
        st.dataframe(pd.DataFrame(np.round(A, 4)).astype(float))
        ev = np.linalg.eigvalsh(A) if np.allclose(A, A.T) else np.linalg.eigvals(A)
        ev_strs = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag) > 1e-10 else f"{v.real:.3f}" for v in ev]
        st.write("Eigenvalues:", "  |  ".join(ev_strs))
    with col2:
        st.write(f"**V** ({d}×{d})")
        st.dataframe(pd.DataFrame(np.round(V, 4)).astype(float))
        ev2 = np.linalg.eigvalsh(V) if np.allclose(V, V.T) else np.linalg.eigvals(V)
        ev_strs2 = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag) > 1e-10 else f"{v.real:.3f}" for v in ev2]
        st.write("Eigenvalues:", "  |  ".join(ev_strs2))

# ──────────────────────────────────────────────────────────────────────────────
# Initialise tokens on S^{d-1}
# ──────────────────────────────────────────────────────────────────────────────
rng_tok = np.random.default_rng(int(st.session_state.token_seed))
X0_raw  = rng_tok.standard_normal((n_tokens, d))
X0      = X0_raw / np.linalg.norm(X0_raw, axis=1, keepdims=True)
x0_flat = X0.ravel()

# ──────────────────────────────────────────────────────────────────────────────
# ODE right-hand side & Integration function
# ──────────────────────────────────────────────────────────────────────────────
def make_rhs(A, V, n, d, beta, is_softmax):
    AT = A.T
    VT = V.T

    def rhs(t, x_flat):
        X = x_flat.reshape(n, d)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)

        scores = (X @ AT) @ X.T

        if is_softmax:
            s = beta * scores
            s -= s.max(axis=1, keepdims=True)
            f_scores = np.exp(s)
            Z = f_scores.sum(axis=1, keepdims=True)
        else:
            f_scores = beta * scores
            Z = float(n)

        XV     = X @ VT
        update = (f_scores / Z) @ XV

        inner = np.sum(X * update, axis=1, keepdims=True)
        dX    = update - inner * X

        return dX.ravel()

    return rhs

t_eval = np.linspace(0.0, T, frames)

def integrate_model(is_softmax, model_label):
    rhs_fn = make_rhs(A, V, n_tokens, d, beta, is_softmax)
    with st.spinner(f"Integrating {n_tokens} tokens × d={d}  |  {case}  |  {model_label}  |  {solver_choice}..."):
        sol_obj = solve_ivp(
            rhs_fn,
            (0.0, T),
            x0_flat,
            method=solver_choice,
            t_eval=t_eval,
            atol=1e-8,
            rtol=1e-6,
            max_step=T / 200.0,
            dense_output=False,
        )
    if not sol_obj.success:
        st.error(f"ODE solver failed for {model_label}: {sol_obj.message}")
        st.stop()
    
    X_tr = sol_obj.y.reshape(n_tokens, d, frames)
    norms_tr = np.linalg.norm(X_tr, axis=1, keepdims=True)
    X_tr = X_tr / np.maximum(norms_tr, 1e-12)
    return sol_obj, X_tr

# Integrate based on selections
if compare_fig1:
    sol_lin, X_traj_lin = integrate_model(False, "Linear")
    sol_soft, X_traj_soft = integrate_model(True, "Softmax")
    
    # Rest of the app defaults to Linear as requested
    sol = sol_lin
    X_traj = X_traj_lin
    label = "Linear"
    
    stiff = sol_lin.nfev > 50_000 or sol_soft.nfev > 50_000
    nfev_msg = f"✅ Done — Linear: **{sol_lin.nfev}** evals, Softmax: **{sol_soft.nfev}** evals."
    max_norm_dev = max(
        np.max(np.abs(np.linalg.norm(X_traj_lin, axis=1) - 1.0)),
        np.max(np.abs(np.linalg.norm(X_traj_soft, axis=1) - 1.0))
    )
else:
    label  = "Softmax" if use_softmax else "Linear"
    sol, X_traj = integrate_model(use_softmax, label)
    
    stiff = sol.nfev > 50_000
    nfev_msg = f"✅ Done — **{sol.nfev}** function evaluations."
    max_norm_dev = np.max(np.abs(np.linalg.norm(X_traj, axis=1) - 1.0))

st.success(
    nfev_msg + ("  ⚠️ High eval count — consider Radau." if stiff else "  No stiffness issues.")
)
st.info(f"Max deviation from unit sphere: **{max_norm_dev:.2e}**")

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Token dynamics")

def calc_similarities(X_tr):
    idx_i, idx_j    = np.triu_indices(n_tokens, k=1)
    c_sim_time      = np.zeros((len(idx_i), frames))
    c2_sim_time     = np.zeros((len(idx_i), frames))
    for t_idx in range(frames):
        Xt = X_tr[:, :, t_idx]
        G  = Xt @ Xt.T
        c_sim_time[:,  t_idx] = G[idx_i, idx_j]
        c2_sim_time[:, t_idx] = G[idx_i, idx_j]**2
    return c_sim_time, c2_sim_time

# Calculate metrics for the primary trajectory (used for Fig 1, Fig 2, and Fig 3)
cos_sim_time, cos2_sim_time = calc_similarities(X_traj)
mean_cos  = cos_sim_time.mean(axis=0)
std_cos   = cos_sim_time.std(axis=0)
mean_cos2 = cos2_sim_time.mean(axis=0)
std_cos2  = cos2_sim_time.std(axis=0)

if compare_fig1:
    # Calculate additional metrics for Softmax solely for Fig 1
    cos_sim_time_soft, cos2_sim_time_soft = calc_similarities(X_traj_soft)
    mean_cos_soft  = cos_sim_time_soft.mean(axis=0)
    mean_cos2_soft = cos2_sim_time_soft.mean(axis=0)

fig1, axes = plt.subplots(1, 1, figsize=(20, 11))
ax = axes

if compare_fig1:
    # Plot Linear
    ax.plot(sol.t, mean_cos2, color="blue", lw=7.0, linestyle='-', label=r"$\mathcal{\hat{R}}_2(t)$ (linear)")
    ax.plot(sol.t, mean_cos, color="royalblue", lw=5.5, linestyle='--', alpha=0.8, label=r"$\mathcal{\hat{R}}_1(t)$ (linear)")
    
    # Plot Softmax
    ax.plot(sol_soft.t, mean_cos2_soft, color="red", lw=7.0, linestyle='-.', label=r"$\mathcal{\hat{R}}_2(t)$ (softmax)")
    ax.plot(sol_soft.t, mean_cos_soft, color="orangered", lw=5.5, linestyle=':', alpha=0.8, label=r"$\mathcal{\hat{R}}_1(t)$ (softmax)")
    
    ax.set_title("Linear vs Softmax Attention", fontsize=26)
else:
    # Plot primary selected model only
    ax.plot(sol.t, mean_cos2, color="tomato", lw=7.0, label=r"$\mathcal{\hat{R}}_2(t)$")
    ax.plot(sol.t, mean_cos, color="steelblue", lw=5.5, linestyle='--', alpha=0.8, label=r"$\mathcal{\hat{R}}_1(t)$")
    ax.set_title(f"{label} Attention", fontsize=24)

ax.axhline(1.0, color='gray', lw=3.0, linestyle=':')
ax.axhline(0.0, color='gray',  lw=3.0, linestyle=':')

ax.set_xlabel("t", fontsize=32)
# ax.set_ylabel(r"$\mathcal{\hat{R}}_2(t)$", fontsize=24)
ax.set_xlim(0, T)
ax.set_ylim(-0.1, 1.05)
ax.tick_params(axis='both', which='major', labelsize=32)
ax.legend(fontsize=36, loc='best')
ax.grid(alpha=0.3)
ax.xaxis.get_offset_text().set_fontsize(32)

plt.tight_layout()
st.pyplot(fig1)

if compare_fig2:
    figd1, axd1 = plt.subplots(1, 1, figsize=(20, 11))

    dim_means = X_traj_lin.mean(axis=0)
    cmap = plt.cm.tab10
    for dim_idx in range(min(d, 10)):
        axd1.plot(sol.t, dim_means[dim_idx], color=cmap(dim_idx % 10),
                lw=6.0, label=f"dim {dim_idx}")
    axd1.set_xlabel("t", fontsize=32)
    axd1.set_ylabel("mean token value", fontsize=32)
    axd1.set_title("Per-dimension mean across tokens (Linear)", fontsize=26)
    axd1.set_xlim(0, T)
    axd1.tick_params(axis='both', which='major', labelsize=32)
    axd1.grid(alpha=0.3)
    # if d <= 10:
    #     axd1.legend(fontsize=24, ncol=2)

    plt.tight_layout()
    st.pyplot(figd1)

    figd2, axd2 = plt.subplots(1, 1, figsize=(20, 11))

    dim_means = X_traj_soft.mean(axis=0)
    cmap = plt.cm.tab10
    for dim_idx in range(min(d, 10)):
        axd2.plot(sol_soft.t, dim_means[dim_idx], color=cmap(dim_idx % 10),
                lw=6.0, label=f"dim {dim_idx}")
    axd2.set_xlabel("t", fontsize=32)
    axd2.set_ylabel("mean token value", fontsize=32)
    axd2.set_title("Per-dimension mean across tokens (Softmax)", fontsize=26)
    axd2.set_xlim(0, T)
    axd2.tick_params(axis='both', which='major', labelsize=32)
    axd2.grid(alpha=0.3)
    # if d <= 10:
    #     axd2.legend(fontsize=24, ncol=2)

    plt.tight_layout()
    st.pyplot(figd2)

# All further outputs continue relying solely on the selected primary 'X_traj' and 'cos2_sim_time'
with st.expander("Pairwise cosine similarity distribution  (t=0 vs t=T)"):
    vals_t0 = cos2_sim_time[:, 0]
    vals_tT = cos2_sim_time[:, -1]

    spike_t0 = np.mean(vals_t0 > 0.999)
    spike_tT = np.mean(vals_tT > 0.999)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax_h, vals, t_label, color, spike in [
        (axes2[0], vals_t0, "t=0",   "steelblue", spike_t0),
        (axes2[1], vals_tT, f"t={T}", "tomato",    spike_tT),
    ]:
        bins = np.linspace(0, 1, 42)
        interior = vals[vals < 0.999]
        at_one   = vals[vals >= 0.999]

        if len(interior) > 0:
            ax_h.hist(interior, bins=bins[:-1], alpha=0.7, color=color, label="spread pairs")
        if len(at_one) > 0:
            bar_width = bins[1] - bins[0]
            ax_h.bar(1.0, len(at_one), width=bar_width, align="edge",
                     color="green", alpha=0.8, label=f"clustered pairs: {spike*100:.1f}%")

        ax_h.axvline(1.0, color="green", lw=2.5, linestyle="--")
        ax_h.set_xlabel(r"$\langle x_i,x_j\rangle^2$", fontsize=14)
        ax_h.set_ylabel("Count", fontsize=14)
        ax_h.set_xlim(-0.02, 1.08)
        ax_h.set_title(f"Distribution at {t_label}", fontsize=18)
        ax_h.tick_params(axis='both', which='major', labelsize=12)
        ax_h.legend(fontsize=14)
        ax_h.grid(alpha=0.3)

    plt.suptitle(r"Squared cosine similarity distribution: $t=0$ vs $t=T$", fontsize=22, y=1.05)
    plt.tight_layout()
    st.pyplot(fig2)

with st.expander("Cosine similarity matrices  (t=0 and t=T, cluster-sorted)"):
    if compare_fig1:
        X_final = X_traj[:, :, -1]
        _, _, Vt = np.linalg.svd(X_final, full_matrices=False)
        dominant    = Vt[0]
        projections = X_final @ dominant
        sort_order  = np.argsort(-projections)
        n_plus      = int((projections >= 0).sum())

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
            Xt = X_traj[:, :, t_idx]
            G  = Xt @ Xt.T
            score_mat = G[np.ix_(sort_order, sort_order)]

            mat_std  = score_mat.std()
            mat_mean = score_mat.mean()
            uniformity = mat_std / (abs(mat_mean) + 1e-12)

            if uniformity < 1e-3:
                from matplotlib.colors import LinearSegmentedColormap
                red_cmap = LinearSegmentedColormap.from_list("solid_red", ["#8D0226", "#5C001F"])
                d_val = abs(mat_mean) * 0.01 + 1e-9
                im = axes3[ax_idx].imshow(score_mat, cmap=red_cmap, aspect="auto",
                                        vmin=mat_mean - d_val, vmax=mat_mean + d_val)
                axes3[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=30)    
            else:
                im = axes3[ax_idx].imshow(score_mat, cmap="RdBu_r", aspect="auto",
                                        vmin=-1, vmax=1)
                axes3[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=30)
                if n_tokens <= 25:
                    for row in range(n_tokens):
                        for col in range(n_tokens):
                            v = score_mat[row, col]
                            axes3[ax_idx].text(col, row, f"{v:.2f}", ha="center", va="center",
                                            fontsize=5.0, color="black" if abs(v) < 0.5 else "white")

            # Draw separator line between the two groups
            if 0 < n_plus < n_tokens:
                axes3[ax_idx].axhline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")
                axes3[ax_idx].axvline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")

            axes3[ax_idx].set_xlabel("token j", fontsize=28)
            axes3[ax_idx].set_ylabel("token k", fontsize=28)
            
            # Enforce integer ticks for token axes
            axes3[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes3[ax_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes3[ax_idx].tick_params(axis='both', which='major', labelsize=16)
            
            cbar = plt.colorbar(im, ax=axes3[ax_idx])
            cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        st.pyplot(fig3)

        X_final = X_traj_soft[:, :, -1]
        _, _, Vt = np.linalg.svd(X_final, full_matrices=False)
        dominant    = Vt[0]
        projections = X_final @ dominant
        sort_order  = np.argsort(-projections)
        n_plus      = int((projections >= 0).sum())

        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
            Xt = X_traj_soft[:, :, t_idx]
            G  = Xt @ Xt.T
            score_mat = G[np.ix_(sort_order, sort_order)]

            mat_std  = score_mat.std()
            mat_mean = score_mat.mean()
            uniformity = mat_std / (abs(mat_mean) + 1e-12)

            if uniformity < 1e-3:
                from matplotlib.colors import LinearSegmentedColormap
                red_cmap = LinearSegmentedColormap.from_list("solid_red", ["#8D0226", "#5C001F"])
                d_val = abs(mat_mean) * 0.01 + 1e-9
                im = axes5[ax_idx].imshow(score_mat, cmap=red_cmap, aspect="auto",
                                        vmin=mat_mean - d_val, vmax=mat_mean + d_val)
                axes5[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=26)    
            else:
                im = axes5[ax_idx].imshow(score_mat, cmap="RdBu_r", aspect="auto",
                                        vmin=-1, vmax=1)
                axes5[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=26)
                if n_tokens <= 25:
                    for row in range(n_tokens):
                        for col in range(n_tokens):
                            v = score_mat[row, col]
                            axes5[ax_idx].text(col, row, f"{v:.2f}", ha="center", va="center",
                                            fontsize=5.0, color="black" if abs(v) < 0.5 else "white")

            # Draw separator line between the two groups
            if 0 < n_plus < n_tokens:
                axes5[ax_idx].axhline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")
                axes5[ax_idx].axvline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")

            axes5[ax_idx].set_xlabel("token j", fontsize=24)
            axes5[ax_idx].set_ylabel("token k", fontsize=24)
            
            # Enforce integer ticks for token axes
            axes5[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes5[ax_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes5[ax_idx].tick_params(axis='both', which='major', labelsize=15)
            
            cbar = plt.colorbar(im, ax=axes5[ax_idx])
            cbar.ax.tick_params(labelsize=15)

        plt.tight_layout()
        st.pyplot(fig5)
    else:
        X_final = X_traj[:, :, -1]
        _, _, Vt = np.linalg.svd(X_final, full_matrices=False)
        dominant    = Vt[0]
        projections = X_final @ dominant
        sort_order  = np.argsort(-projections)
        n_plus      = int((projections >= 0).sum())

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
            Xt = X_traj[:, :, t_idx]
            G  = Xt @ Xt.T
            score_mat = G[np.ix_(sort_order, sort_order)]

            mat_std  = score_mat.std()
            mat_mean = score_mat.mean()
            uniformity = mat_std / (abs(mat_mean) + 1e-12)

            if uniformity < 1e-3:
                from matplotlib.colors import LinearSegmentedColormap
                red_cmap = LinearSegmentedColormap.from_list("solid_red", ["#8D0226", "#5C001F"])
                d_val = abs(mat_mean) * 0.01 + 1e-9
                im = axes3[ax_idx].imshow(score_mat, cmap=red_cmap, aspect="auto",
                                        vmin=mat_mean - d_val, vmax=mat_mean + d_val)
                axes3[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=26)    
            else:
                im = axes3[ax_idx].imshow(score_mat, cmap="RdBu_r", aspect="auto",
                                        vmin=-1, vmax=1)
                axes3[ax_idx].set_title(r"$\langle x_j, x_k \rangle$ (" + t_label + ")", fontsize=26)
                if n_tokens <= 25:
                    for row in range(n_tokens):
                        for col in range(n_tokens):
                            v = score_mat[row, col]
                            axes3[ax_idx].text(col, row, f"{v:.2f}", ha="center", va="center",
                                            fontsize=5.0, color="black" if abs(v) < 0.5 else "white")

            # Draw separator line between the two groups
            if 0 < n_plus < n_tokens:
                axes3[ax_idx].axhline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")
                axes3[ax_idx].axvline(n_plus - 0.5, color="gold", lw=3.0, linestyle="--")

            axes3[ax_idx].set_xlabel("token j", fontsize=24)
            axes3[ax_idx].set_ylabel("token k", fontsize=24)
            
            # Enforce integer ticks for token axes
            axes3[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes3[ax_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes3[ax_idx].tick_params(axis='both', which='major', labelsize=17)
            
            cbar = plt.colorbar(im, ax=axes3[ax_idx])
            cbar.ax.tick_params(labelsize=17)

        plt.tight_layout()
        st.pyplot(fig3)

with st.expander("Token positions at t=0 and t=T"):
    if compare_fig1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Linear Attention: t=0**")
            st.dataframe(pd.DataFrame(np.round(X_traj[:, :, 0].astype(float), 4)))
        with col2:
            st.write(f"**Linear Attention: t={T}**")
            st.dataframe(pd.DataFrame(np.round(X_traj[:, :, -1].astype(float), 4)))
        st.markdown("**Softmax Attention**")
        col3, col4 = st.columns(2)
        with col3:
            st.write("**t=0**")
            st.dataframe(pd.DataFrame(np.round(X_traj_soft[:, :, 0].astype(float), 4)))
        with col4:
            st.write(f"**t={T}**")
            st.dataframe(pd.DataFrame(np.round(X_traj_soft[:, :, -1].astype(float), 4)))
        st.write(f"Trajectory shape: `(n_tokens={n_tokens}, d={d}, frames={frames})`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**t=0**")
            st.dataframe(pd.DataFrame(np.round(X_traj[:, :, 0].astype(float), 4)))
        with col2:
            st.write(f"**t={T}**")
            st.dataframe(pd.DataFrame(np.round(X_traj[:, :, -1].astype(float), 4)))
        st.write(f"Trajectory shape: `(n_tokens={n_tokens}, d={d}, frames={frames})`")