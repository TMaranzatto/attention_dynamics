"""
Attention Dynamics on S^{d-1}
==============================

Integrates the continuous-time self-attention ODE on the unit sphere:

  Linear:   dx_k/dt = P^perp_{x_k} ( beta/Z_k  sum_j <Qx_k, Kx_j> V x_j )
  Softmax:  dx_k/dt = P^perp_{x_k} ( 1/Z_k     sum_j exp(beta <Qx_k, Kx_j>) V x_j )

where:
  - x_k in S^{d-1}  (unit sphere)
  - P^perp_x y = y - <x, y> x  (projection onto tangent space at x)
  - Z_k = sum_j <Qx_k, Kx_j>          (linear, raw sum)
        or n                             (linear, uniform)
        or sum_j exp(beta <Qx_k, Kx_j>) (softmax, always positive)
  - Q, K, V are d x d real matrices;i A = Q^T K is the combined query-key matrx

Tokens are initialised uniformly on S^{d-1} via normalised Gaussian draws.
ODE solver: RK45 (default), DOP853, or Radau — selectable in sidebar.

More details about ODE solvers and stiffness:
- RK45: Explicit Runge-Kutta method, good for non-stiff problems. Fast but may struggle with stiff dynamics, leading to many function evaluations or failure.
- DOP853: Higher-order explicit method, can be more efficient for smooth problems but still struggles with stiffness.
- Radau: Implicit method designed for stiff problems. More robust when stiffness is present but computationally heavier.

Stiffness can arise if tokens rapidly cluster together, causing large gradients and requiring very small time steps for explicit solvers. If you see warnings about excessive function evaluations or suspect stiffness, try switching to Radau for more reliable integration.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Dynamics on Sphere", layout="wide")
st.title("Attention Token Dynamics on $\\mathbb{S}^{d-1}$")
st.markdown(r"""
Integrates the continuous-time self-attention ODE on the unit sphere $\mathbb{S}^{d-1}$:

$$\dot{x}_k = P^\perp_{x_k}\!\left(\frac{1}{Z_k}\sum_{j=1}^n f\!\left(\beta\langle Qx_k, Kx_j\rangle\right) V x_j\right)$$

with $P^\perp_x y = y - \langle x, y\rangle x$,  and $f(s) = s$ (linear) or $f(s) = e^s$ (softmax).
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
    attn_type = st.radio("Attention type", ["Linear  f(s) = s", "Softmax  f(s) = exp(s)"])
    use_softmax = attn_type.startswith("Softmax")

    beta = st.slider("β (inverse temperature)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)

    if not use_softmax:
        norm_type = st.radio(
            "Normalisation  $Z_k$  (linear only)",
            ["Uniform  n", "Raw sum  Σ⟨Qxₖ, Kxⱼ⟩"]
        )
        use_raw_norm = norm_type.startswith("Raw")
    else:
        use_raw_norm = False   # softmax always uses its own natural Z_k

    st.markdown("---")
    st.subheader("Matrix initialisation")
    matrix_scale = st.slider(
        "Entry std-dev σ (for Q, K, V)",
        min_value=0.1, max_value=3.0, value=1.0, step=0.1,
        help="Entries drawn from N(0, σ²/d). σ=1 keeps spectral norm ≈ O(1)."
    )
    matrix_seed = int(st.number_input("Matrix random seed", min_value=0, max_value=10**9, value=42, step=1))

    st.markdown("---")
    st.subheader("Token initialisation")
    if "token_seed" not in st.session_state:
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    if st.button("Randomise initial tokens"):
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    st.write("Token seed:", st.session_state.token_seed)

    st.markdown("---")
    solver_choice = st.selectbox(
        "ODE solver",
        ["RK45", "DOP853", "Radau"],
        index=0,
        help="RK45/DOP853: explicit, fast. Radau: implicit, use if stiffness warnings appear."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Build Q, K, V  and  A = Q^T K
# ──────────────────────────────────────────────────────────────────────────────
rng_mat = np.random.default_rng(int(matrix_seed))
scale   = float(matrix_scale) / np.sqrt(d)

Q = rng_mat.normal(0.0, scale, (d, d))
K = rng_mat.normal(0.0, scale, (d, d))
V = rng_mat.normal(0.0, scale, (d, d))
A = Q.T @ K   # d x d  — the combined query-key matrix

with st.expander("Show Q, K, V matrices and eigenvalues"):
    for name, mat in [("Q", Q), ("K", K), ("V", V), ("A = QᵀK", A)]:
        st.write(f"**{name}** ({d}\u00d7{d})")
        st.dataframe(pd.DataFrame(np.round(mat, 4)).astype(float))
        eigvals = np.linalg.eigvals(mat)
        eigval_strs = [
            f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag) > 1e-10 else f"{v.real:.3f}"
            for v in eigvals
        ]
        # Find the largest-magnitude eigenvalue shown in bold
        max_idx = np.argmax(np.abs(eigvals))
        st.write(f"Largest-magnitude eigenvalue: **{eigval_strs[max_idx]}** (|{eigvals[max_idx]:.3f}| = {abs(eigvals[max_idx]):.3f})")
        st.write("Eigenvalues:", "  |  ".join(eigval_strs))
        st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Initialise tokens uniformly on S^{d-1}
# ──────────────────────────────────────────────────────────────────────────────
rng_tok = np.random.default_rng(int(st.session_state.token_seed))
X0_raw  = rng_tok.standard_normal((n_tokens, d))
X0      = X0_raw / np.linalg.norm(X0_raw, axis=1, keepdims=True)   # unit sphere
x0_flat = X0.ravel()

# ──────────────────────────────────────────────────────────────────────────────
# ODE right-hand side
# ──────────────────────────────────────────────────────────────────────────────
def make_rhs(Q, K, V, n, d, beta, use_softmax, use_raw_norm):
    """
    State: x_flat of shape (n*d,) — row-major flattening of X (n, d).

    For each token k:
        scores_k  = X @ (Q^T K)^T @ x_k = X @ K^T @ Q @ x_k   shape (n,)
                  = x_k^T Q^T K x_j  for each j                (attention scores)
        f_scores  = beta * scores_k       (linear)
                  = exp(beta * scores_k)  (softmax)
        Z_k       = sum(f_scores)         or n (linear uniform)
        update_k  = (1/Z_k) * f_scores @ (X @ V^T)             shape (d,)
        proj_k    = update_k - <x_k, update_k> * x_k           tangent projection
    """
    KtQ  = K.T @ Q      # d x d  precomputed: (Qx_k)^T(Kx_j) = x_k^T Q^T K x_j
    VT   = V.T          # d x d

    def rhs(t, x_flat):
        X = x_flat.reshape(n, d)          # (n, d)

        # Re-normalise onto sphere each step to prevent drift
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)

        XV  = X @ VT                      # (n, d)  — XV[j] = V x_j

        # Attention scores: score[k, j] = x_k^T Q^T K x_j = (KtQ x_k) . x_j
        # = (X @ KtQ^T)[k] . x_j  but we need all k,j pairs
        # scores[k, j] = (X @ KtQ^T @ X^T)[k, j]
        scores = (X @ KtQ) @ X.T         # (n, n)  scores[k,j] = <Qx_k, Kx_j>

        if use_softmax:
            # Subtract max per row for numerical stability
            scores_shifted = beta * scores
            scores_shifted -= scores_shifted.max(axis=1, keepdims=True)
            f_scores = np.exp(scores_shifted)              # (n, n)
            Z = f_scores.sum(axis=1, keepdims=True)        # (n, 1)  always > 0
        else:
            f_scores = beta * scores                       # (n, n)
            if use_raw_norm:
                Z = f_scores.sum(axis=1, keepdims=True)    # (n, 1)  may be near zero
                # Robust guard: if |Z_k| is small relative to the score magnitudes,
                # the ODE becomes singular and stiff. Clamp with a meaningful floor.
                score_scale = np.abs(f_scores).mean() * n  # typical scale of Z
                floor = np.maximum(score_scale * 1e-3, 1e-6)
                # Preserve sign but enforce minimum absolute value
                Z_abs   = np.abs(Z)
                Z_sign  = np.sign(Z)
                Z_sign  = np.where(Z_sign == 0, 1.0, Z_sign)
                Z = Z_sign * np.maximum(Z_abs, floor)
            else:
                Z = n                                      # scalar

        # Weighted sum of value-projected tokens
        # update[k] = (1/Z_k) * sum_j f_scores[k,j] * (V x_j)
        update = (f_scores / Z) @ XV      # (n, d)

        # Project onto tangent space of S^{d-1} at x_k:
        # P^perp_{x_k} y = y - <x_k, y> x_k
        inner  = np.sum(X * update, axis=1, keepdims=True)   # (n, 1)  <x_k, update_k>
        dX     = update - inner * X                           # (n, d)

        return dX.ravel()

    return rhs

rhs_fn = make_rhs(Q, K, V, n_tokens, d, beta, use_softmax, use_raw_norm)

# ──────────────────────────────────────────────────────────────────────────────
# Integrate
# ──────────────────────────────────────────────────────────────────────────────
t_eval = np.linspace(0.0, T, frames)

label = "Softmax" if use_softmax else "Linear"

# For linear + raw-sum normalisation, loosen tolerances to avoid stiffness spiral
if not use_softmax and use_raw_norm:
    atol, rtol = 1e-5, 1e-4
    st.warning(
        "⚠️ Linear attention with raw-sum Z_k can produce near-zero denominators "
        "that make the ODE stiff. Using looser tolerances (atol=1e-5, rtol=1e-4) "
        "and capped step size. Consider switching to **Uniform Z_k = n** or "
        "**Softmax** for well-conditioned dynamics."
    )
else:
    atol, rtol = 1e-8, 1e-6

# max_step prevents the solver from taking steps so small it never finishes
max_step = T / 200.0

with st.spinner(f"Integrating {n_tokens} tokens x d={d} ({label} attention, {solver_choice})..."):
    sol = solve_ivp(
        rhs_fn,
        (0.0, T),
        x0_flat,
        method=solver_choice,
        t_eval=t_eval,
        atol=atol,
        rtol=rtol,
        max_step=max_step,
        dense_output=False,
    )

if not sol.success:
    st.error(f"ODE solver failed: {sol.message}")
    st.stop()

# Unpack: (n*d, frames) → (n, d, frames)
X_traj = sol.y.reshape(n_tokens, d, frames)

# Re-normalise trajectory onto sphere (correct any solver drift)
norms_traj = np.linalg.norm(X_traj, axis=1, keepdims=True)   # (n, 1, frames)
X_traj     = X_traj / np.maximum(norms_traj, 1e-12)

stiff_warning = sol.nfev > 50_000
st.success(
    f"✅ Integration complete — **{sol.nfev}** function evaluations.  "
    + ("⚠️ High eval count — consider switching to Radau if results look wrong." if stiff_warning else "No stiffness issues detected.")
)

# ──────────────────────────────────────────────────────────────────────────────
# Verify sphere constraint
# ──────────────────────────────────────────────────────────────────────────────
norms_check = np.linalg.norm(X_traj, axis=1)   # (n, frames)
max_norm_dev = np.max(np.abs(norms_check - 1.0))
st.info(f"Max deviation from unit sphere ‖xₖ‖=1 over all tokens and times: **{max_norm_dev:.2e}**")

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Token dynamics")

# --- 1. Pairwise cosine similarities over time ---
# cosine sim between token i and j = <x_i, x_j>  (both on unit sphere)
# Track mean and std of all pairwise cosine similarities
fig1, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: mean ± std of pairwise cosine similarities
idx_i, idx_j = np.triu_indices(n_tokens, k=1)
n_pairs = len(idx_i)
cos_sim_time = np.zeros((n_pairs, frames))
for t_idx in range(frames):
    Xt = X_traj[:, :, t_idx]                    # (n, d)
    G  = Xt @ Xt.T                               # (n, n) Gram = cosine sim (unit sphere)
    cos_sim_time[:, t_idx] = G[idx_i, idx_j]

mean_cos = cos_sim_time.mean(axis=0)
std_cos  = cos_sim_time.std(axis=0)

ax = axes[0]
ax.plot(sol.t, mean_cos, color="steelblue", lw=2, label="mean cos sim")
ax.fill_between(sol.t, mean_cos - std_cos, mean_cos + std_cos,
                alpha=0.25, color="steelblue", label="±1 std")
ax.axhline(1.0, color='gray', lw=0.8, linestyle='--', label='perfect clustering')
ax.set_xlabel("t")
ax.set_ylabel("cosine similarity")
ax.set_title("Pairwise cosine similarities over time")
ax.set_xlim(0, T)
ax.set_ylim(-1.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Right: per-dimension mean across tokens
dim_means = X_traj.mean(axis=0)    # (d, frames)
ax2 = axes[1]
cmap = plt.cm.tab10
for dim_idx in range(min(d, 10)):
    ax2.plot(sol.t, dim_means[dim_idx], color=cmap(dim_idx % 10),
             lw=1.5, label=f"dim {dim_idx}")
ax2.set_xlabel("t")
ax2.set_ylabel("mean value")
ax2.set_title("Per-dimension mean across tokens")
ax2.set_xlim(0, T)
ax2.grid(alpha=0.3)
if d <= 10:
    ax2.legend(fontsize=7, ncol=2)

plt.tight_layout()
st.pyplot(fig1)

# --- 2. Clustering: how many distinct clusters? ---
# Track the minimum pairwise cosine similarity — if → 1, full consensus
with st.expander("Show pairwise cosine similarity distribution  (t=0 vs t=T)"):
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(cos_sim_time[:, 0],  bins=40, alpha=0.6, color="steelblue", label="t=0")
    ax3.hist(cos_sim_time[:, -1], bins=40, alpha=0.6, color="tomato",    label=f"t={T}")
    ax3.set_xlabel("Cosine similarity ⟨xᵢ, xⱼ⟩")
    ax3.set_ylabel("Count")
    ax3.set_title("Pairwise cosine similarities: t=0 vs t=T")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig2)

# --- 3. Attention score matrix at t=0 and t=T ---
with st.expander("Show attention score matrices  (t=0 and t=T)"):
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
        Xt    = X_traj[:, :, t_idx]
        KtQ   = K.T @ Q
        raw   = (Xt @ KtQ) @ Xt.T              # (n, n) raw scores
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

# --- 4. Raw trajectory data ---
with st.expander("Show token positions at t=0 and t=T"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Token positions at t=0**  (each row = one token)")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, 0].astype(float), 4)))
    with col2:
        st.write(f"**Token positions at t={T}**")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, -1].astype(float), 4)))
    st.write(f"Trajectory array shape: `(n_tokens={n_tokens}, d={d}, frames={frames})`")