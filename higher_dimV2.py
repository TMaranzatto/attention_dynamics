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
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Dynamics on Sphere", layout="wide")
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

    st.markdown("---")
    attn_type   = st.radio("Attention type", ["Softmax  f(s) = exp(s)", "Linear  f(s) = s"])
    use_softmax = attn_type.startswith("Softmax")
    beta        = st.slider("β (inverse temperature)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    if not use_softmax:
        st.info("Linear attention: Z_k = n (uniform normalisation).")

    st.markdown("---")
    st.subheader("Matrix case")
    case = st.radio(
        "Choose case",
        ["Case 1: V=I, A sym PD",
         "Case 2: A=I, V symmetric",
         "Case 4: A=I, V Hamiltonian",
         "Random"]
    )
    matrix_seed = int(st.number_input("Matrix random seed", min_value=0, max_value=10**9, value=42, step=1))
    rng_mat     = np.random.default_rng(int(matrix_seed))

    st.markdown("---")

    # ── Case-specific parameters ──────────────────────────────────────────────
    if case == "Case 1: V=I, A sym PD":
        st.markdown(r"""
**Case 1**: $V = I$, $A$ symmetric positive definite.

Clustering is governed by definiteness of $A + A^\top = 2A$.
Since $A$ is symmetric PD here, clustering is guaranteed for linear attention.
        """)
        a_scale = st.slider("Eigenvalue scale σ", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                            help="Eigenvalues of A drawn from Uniform(0.1, σ).")
        Q_orth   = ortho_group.rvs(d, random_state=int(matrix_seed))
        eigvals_A = rng_mat.uniform(0.1, float(a_scale), size=d)
        A = Q_orth @ np.diag(eigvals_A) @ Q_orth.T
        V = np.eye(d)
        st.write("A eigenvalues:", np.round(np.sort(eigvals_A)[::-1], 3).tolist())

    elif case == "Case 2: A=I, V symmetric":
        st.markdown(r"""
**Case 2**: $A = I$, $V$ symmetric.

Clustering governed by the **top eigenvalue** $\lambda_1$ of $V$.
Positive $\lambda_1$ → clustering; negative → dispersion.
        """)
        top_eig     = st.slider("Top eigenvalue λ₁ of V", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        other_scale = st.slider("Other eigenvalues scale", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        Q_orth      = ortho_group.rvs(d, random_state=int(matrix_seed))
        eigvals_V   = rng_mat.normal(0.0, float(other_scale), size=d)
        eigvals_V[0] = float(top_eig)
        A = np.eye(d)
        V = Q_orth @ np.diag(eigvals_V) @ Q_orth.T
        st.write("V eigenvalues:", np.round(np.sort(eigvals_V)[::-1], 3).tolist())

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
            V[d-1, d-1] = a_param

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
# ODE right-hand side
# ──────────────────────────────────────────────────────────────────────────────
def make_rhs(A, V, n, d, beta, use_softmax):
    """
    dx_k/dt = P^perp_{x_k} ( 1/Z_k  sum_j f(beta <Ax_k, x_j>) V x_j )

    scores[k,j] = <Ax_k, x_j> = (X @ A.T @ X.T)[k,j]
    f_scores    = beta * scores           (linear,  Z = n)
                = exp(beta * scores)      (softmax, Z = row sum)
    update[k]   = (1/Z_k) sum_j f_scores[k,j] (V x_j)
    dX[k]       = update[k] - <x_k, update[k]> x_k
    """
    AT = A.T
    VT = V.T

    def rhs(t, x_flat):
        X = x_flat.reshape(n, d)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)

        scores = (X @ AT) @ X.T              # (n, n)  scores[k,j] = <Ax_k, x_j>

        if use_softmax:
            s = beta * scores
            s -= s.max(axis=1, keepdims=True)
            f_scores = np.exp(s)
            Z = f_scores.sum(axis=1, keepdims=True)
        else:
            f_scores = beta * scores
            Z = float(n)

        XV     = X @ VT                      # (n, d)  V x_j for each j
        update = (f_scores / Z) @ XV         # (n, d)

        inner = np.sum(X * update, axis=1, keepdims=True)
        dX    = update - inner * X

        return dX.ravel()

    return rhs

rhs_fn = make_rhs(A, V, n_tokens, d, beta, use_softmax)

# ──────────────────────────────────────────────────────────────────────────────
# Integrate
# ──────────────────────────────────────────────────────────────────────────────
t_eval = np.linspace(0.0, T, frames)
label  = "Softmax" if use_softmax else "Linear"

with st.spinner(f"Integrating {n_tokens} tokens × d={d}  |  {case}  |  {label}  |  {solver_choice}..."):
    sol = solve_ivp(
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

if not sol.success:
    st.error(f"ODE solver failed: {sol.message}")
    st.stop()

X_traj     = sol.y.reshape(n_tokens, d, frames)
norms_traj = np.linalg.norm(X_traj, axis=1, keepdims=True)
X_traj     = X_traj / np.maximum(norms_traj, 1e-12)

stiff = sol.nfev > 50_000
st.success(
    f"✅ Done — **{sol.nfev}** function evaluations.  "
    + ("⚠️ High eval count — consider Radau." if stiff else "No stiffness issues.")
)
max_norm_dev = np.max(np.abs(np.linalg.norm(X_traj, axis=1) - 1.0))
st.info(f"Max deviation from unit sphere: **{max_norm_dev:.2e}**")

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Token dynamics")

# Clustering metric: <x_i, x_j>^2  (squared cosine similarity)
# This is the correct analogue of the 2D order parameter e^{2i*theta}:
# tokens at theta and theta+pi are antipodal (cos sim = -1) but represent
# the same cluster, so we use cos^2 which equals 1 for both aligned and antipodal.
# In higher dims: <x_i,x_j>^2 = ||x_i x_i^T - x_j x_j^T||^2 proxy (Frobenius).
idx_i, idx_j    = np.triu_indices(n_tokens, k=1)
cos_sim_time    = np.zeros((len(idx_i), frames))   # raw <x_i,x_j>
cos2_sim_time   = np.zeros((len(idx_i), frames))   # squared <x_i,x_j>^2
for t_idx in range(frames):
    Xt = X_traj[:, :, t_idx]
    G  = Xt @ Xt.T                          # Gram matrix, entries = <x_i,x_j>
    cos_sim_time[:,  t_idx] = G[idx_i, idx_j]
    cos2_sim_time[:, t_idx] = G[idx_i, idx_j]**2

mean_cos  = cos_sim_time.mean(axis=0)
std_cos   = cos_sim_time.std(axis=0)
mean_cos2 = cos2_sim_time.mean(axis=0)
std_cos2  = cos2_sim_time.std(axis=0)

fig1, ax = plt.subplots(figsize=(7, 4))

# Plot squared cosine similarity (correct clustering metric)
ax.plot(sol.t, mean_cos2, color="tomato", lw=2, label=r"mean $\langle x_i,x_j\rangle^2$ (clustering)")
ax.fill_between(sol.t, mean_cos2 - std_cos2, mean_cos2 + std_cos2,
                alpha=0.2, color="tomato", label="±1 std")
# Also show raw cosine sim as thin dashed for reference
ax.plot(sol.t, mean_cos, color="steelblue", lw=1, linestyle='--',
        alpha=0.6, label=r"mean $\langle x_i,x_j\rangle$ (raw, for ref)")
ax.axhline(1.0, color='green', lw=0.8, linestyle='--', label='full clustering = 1')
ax.axhline(0.0, color='gray',  lw=0.8, linestyle=':')
ax.set_xlabel("t")
ax.set_ylabel(r"$\langle x_i, x_j\rangle^2$")
ax.set_title(f"Clustering metric — {case}")
ax.set_xlim(0, T)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    r"""
#### What this graph shows and why we plot it this way

**The naive approach — and why it fails.**
The most natural way to measure whether tokens are clustering is to track the raw
pairwise cosine similarity $\langle x_i, x_j \rangle$ between every pair of tokens.
If all tokens collapse to the same point on $\mathbb{S}^{d-1}$, every pairwise cosine
similarity equals $1$, and the mean would rise to $1$ over time.

However, this fails for these dynamics. The continuous-time attention ODE does **not**
push all tokens to the same point — it pushes them to cluster in the sense of the
second-order parameter $R_2$. In the 2D case ($d=2$, tokens on $\mathbb{S}^1$), the
order parameter tracked by the OA reduction is

$$R_2(t) = \frac{1}{n} \sum_{j=1}^n e^{2i\theta_j},$$

not $R_1 = \frac{1}{n}\sum e^{i\theta_j}$. The factor of $2$ in the exponent means
the dynamics are $\pi$-periodic in $\theta$: a token at angle $\theta$ and a token at
$\theta + \pi$ are treated as **identical** by $R_2$, even though as vectors on
$\mathbb{S}^1$ they are antipodal ($\langle x_i, x_j \rangle = -1$). Full clustering
in the $R_2$ sense means tokens split into groups at $\theta^*$ and $\theta^* + \pi$,
so the mean raw cosine similarity is exactly $0$ even when the system is perfectly
clustered. This is precisely why earlier plots showed a flat line at $0$ regardless
of parameters.

**The correct metric.**
The right analogue of $|R_2| \to 1$ in dimension $d$ is to measure similarity between
the rank-1 projection matrices $x_i x_i^\top$ rather than between the vectors $x_i$
themselves. The Frobenius inner product between two such projections is

$$\langle x_i x_i^\top,\, x_j x_j^\top \rangle_F = \mathrm{tr}(x_i x_i^\top x_j x_j^\top) = \langle x_i, x_j \rangle^2.$$

This equals $1$ whether $x_i = x_j$ (aligned) or $x_i = -x_j$ (antipodal), and equals
$0$ when the tokens are orthogonal. It is invariant under the sign flip $x \mapsto -x$
that the dynamics treat as equivalent.

**What to look for on the graph.**
- $\langle x_i, x_j \rangle^2 \to 1$: tokens are clustering (all aligning or forming
  antipodal pairs), consistent with $|R_2| \to 1$.
- $\langle x_i, x_j \rangle^2 \approx 1/d$: tokens remain approximately uniformly
  spread on $\mathbb{S}^{d-1}$ — no clustering.
- Oscillating $\langle x_i, x_j \rangle^2$: cyclic / Hamiltonian behavior
  (Case 4 with $a < b$).

The shaded band is $\pm 1$ standard deviation across all $\binom{n}{2}$ pairs.
A narrow band near $1$ means tight single-cluster behavior; a wide band means tokens
have split into multiple distinct clusters. The dashed blue line shows raw cosine
similarity for reference — note how it stays near $0$ even when the squared version
is near $1$, which is why we cannot use it as a clustering diagnostic here.
    """
)

with st.expander("Pairwise cosine similarity distribution  (t=0 vs t=T)"):
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(cos2_sim_time[:, 0],  bins=40, alpha=0.6, color="steelblue", label="t=0")
    ax3.hist(cos2_sim_time[:, -1], bins=40, alpha=0.6, color="tomato",    label=f"t={T}")
    ax3.axvline(1.0, color='green', lw=1, linestyle='--', label='full clustering = 1')
    ax3.set_xlabel(r"Squared cosine similarity $\langle x_i,x_j\rangle^2$")
    ax3.set_ylabel("Count")
    ax3.set_title("Clustering metric distribution: t=0 vs t=T")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig2)

with st.expander("Attention score matrices  (t=0 and t=T)"):
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
        Xt  = X_traj[:, :, t_idx]
        raw = (Xt @ A.T) @ Xt.T
        if use_softmax:
            raw_b = beta * raw
            raw_b -= raw_b.max(axis=1, keepdims=True)
            score_mat = np.exp(raw_b)
            score_mat /= score_mat.sum(axis=1, keepdims=True)
            title = f"Softmax attention weights ({t_label})"
        else:
            score_mat = beta * raw
            title = f"Linear attention scores ({t_label})"
        im = axes3[ax_idx].imshow(score_mat, cmap="viridis", aspect="auto")
        axes3[ax_idx].set_title(title)
        axes3[ax_idx].set_xlabel("token j")
        axes3[ax_idx].set_ylabel("token k")
        plt.colorbar(im, ax=axes3[ax_idx])
    plt.tight_layout()
    st.pyplot(fig3)

with st.expander("Token positions at t=0 and t=T"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**t=0**")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, 0].astype(float), 4)))
    with col2:
        st.write(f"**t={T}**")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, -1].astype(float), 4)))
    st.write(f"Trajectory shape: `(n_tokens={n_tokens}, d={d}, frames={frames})`")