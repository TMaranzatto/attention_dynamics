# Attention Dynamics: Token Clustering via Ott-Antonsen Analysis

This repository contains numerical simulations and analysis of attention dynamics in transformer models, focusing on how tokens (represented as points on high-dimensional spheres) cluster together over time. The project implements the theoretical framework from the associated paper using Ott-Antonsen reduction and Watanabe-Strogatz (WS) dynamics.

**Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

The core investigation studies the clustering behavior of $n$ tokens initialized on a sphere $\mathbb{S}^{d-1}$ under the continuous-time self-attention update equations:

$$\dot{x}_k(t) = P^\perp_{x_k(t)}\left(\frac{1}{n}\sum_{j=1}^n h\left(\beta\langle Ax_k(t), x_j(t)\rangle\right) V x_j(t)\right), \quad k = 1,\dots,n$$

where:
- $x_k \in \mathbb{S}^{d-1}$ are token embeddings on the unit sphere
- $P^\perp_{x_k}(\cdot)$ projects onto the tangent space perpendicular to $x_k$
- $A$ and $V$ are $d \times d$ matrices controlling attention dynamics
- $\beta$ is an inverse temperature parameter (rescales time; we fix $\beta = 1$ by default)
- Two attention models are studied:
  - **Linear Self-Attention (LSA)**: $h(y) = y$
  - **Unnormalized Softmax Attention (USA)**: $h(y) = \exp(y)$

The mathematical analysis uses the **Ott-Antonsen (OA) reduction** to predict asymptotic clustering behavior in terms of mean-field parameters $(\rho, \phi)$ representing cluster coherence and phase. Experiments validate theoretical predictions in high dimensions ($d \gg 2$) and compare LSA and USA models across different parameter regimes.

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib, NetworkX, Streamlit

### Setup

```bash
pip install numpy scipy matplotlib networkx streamlit
```

## Project Structure

### Main Scripts

**`watanabe.py`** — Primary simulation engine
- Replicates all main numerical experiments from the paper
- Integrates token trajectories using scipy's `solve_ivp` with RK45 method
- Generates phase portraits for Ott-Antonsen dynamics
- Produces clustering diagnostic plots
- Includes interactive visualization through Streamlit
- Run with: `streamlit run watanabe.py`

**`circle_dynamics.py`** — 2D circle analysis
- Studies token dynamics on $\mathbb{S}^1$ (unit circle)
- Validates OA reduction in 2D case
- Computes equilibrium clustering states

**`sphere_dynamics.py`** — 3D sphere analysis
- Extends analysis to $\mathbb{S}^2$ (unit sphere in 3D)
- Tests robustness of OA predictions in 3D

**`higher_dim.py` & `higher_dimV2.py`** — Higher-dimensional cases
- Studies token clustering in dimension $d > 2$
- Tests spectral construction methods for matrices $A$ and $V$
- Validates dimension-independent bifurcation conditions

**`symbolic_generate_fields.py`** — Symbolic computation
- Uses SymPy to analytically compute Ott-Antonsen reduced dynamics
- Generates field formulas for phase portraits
- Validates theoretical predictions against numerics

**`merged_paper_figures.py`** — Figure generation
- Reproduces all figures from the paper
- Combines multiple experimental configurations

## Numerical Methods

### ODE Integration

All token trajectories are integrated using **SciPy's `solve_ivp` solver** with the following configuration:

- **Method**: RK45 (Runge-Kutta order 4/5)
  - Error controlled by fourth-order estimate
  - Steps performed at fifth-order for accuracy
- **Tolerances**:
  - Absolute tolerance: `atol = 1e-8`
  - Relative tolerance: `rtol = 1e-6`
- **Initial conditions**: Tokens uniformly distributed on $\mathbb{S}^{d-1}$ via normalized Gaussian draws

### Phase Portrait Computation

Ott-Antonsen phase portraits in the $(\rho, \phi)$ plane are computed on a **100 × 100 grid**:
- Domain: $\rho \in [0.05, 1.0]$, $\phi \in [-\pi, \pi]$
- Grid points uniformly spaced; $\rho_{\min} = 0.05$ avoids division-by-zero singularity at $\rho = 0$
- Velocity vectors computed at each grid point
- Visualization via matplotlib's `streamplot` (shows flow direction only)

### Clustering Diagnostics

To quantify token clustering behavior, we track multiple complementary metrics:

**Primary Clustering Metrics:**

**$\hat{\mathcal{R}}_2(t)$ — Mean-squared cosine similarity** (primary diagnostic)

$$\hat{\mathcal{R}}_2(t) := \frac{1}{\binom{n}{2}} \sum_{j < k} \langle x_j(t), x_k(t)\rangle^2 \in [0,1]$$

- Equals 1 when all tokens collapse to antipodal points $\{{x^\star, -x^\star\}}$ (allows unequal cluster sizes)
- Includes complete consensus ($x_1 = x_2 = \cdots = x_n = x^*$)
- Insensitive to sign: $x^{\star}$ and $-{x^{\star}}$ both yield $\hat{\mathcal{R}}_2(t) = 1$

**$\hat{\mathcal{R}}_1(t)$ — Mean cosine similarity** (distinguishes cluster configurations)

$$\hat{\mathcal{R}}_1(t) := \frac{1}{\binom{n}{2}} \sum_{j < k} \langle x_j(t), x_k(t)\rangle \in [0,1]$$

- Equals 1 only if all tokens collapse to a single point $x^*$
- Distinguishes **single-point consensus** ($\hat{\mathcal{R}}_1(t) = 1, \hat{\mathcal{R}}_2(t) = 1$) from **antipodal clustering** ($\hat{\mathcal{R}}_1(t) \approx 0, \hat{\mathcal{R}}_2(t) = 1$)

**Additional Diagnostics:**

**Gram Matrix** $G(t)$ with entries $G_{ij}(t) = \langle x_i(t), x_j(t)\rangle$
- Visualizes token configurations: block structure indicates antipodal clusters
- Positive diagonal/off-diagonal blocks = one cluster; mixed signs = two clusters

**Coordinate-wise Mean** for cyclicality detection

$$m_l(t) := \frac{1}{n}\sum_{j=1}^n x_j^{(l)}(t), \quad l=1,\dots,d$$

- Detects periodic oscillations: convergence of $m(t)$ indicates convergence of clusters
- Non-convergent $m(t)$ with clustered $\hat{\mathcal{R}}_2(t)$ indicates rotating/oscillating clusters

**Interpretation Guide:**

| Configuration | $\hat{\mathcal{R}}_1(t)$ | $\hat{\mathcal{R}}_2(t)$ | $m(t)$ Behavior | Cluster Type |
|---|---|---|---|---|
| No clustering | $\approx 0$ | $\approx 0$ | Converges | Uniform distribution |
| Antipodal clusters | $\approx 0$ | $\to 1$ | Converges | Two opposite poles |
| Single point | $\to 1$ | $\to 1$ | Converges | All tokens at $x^*$ |
| Rotating antipodal | $\approx 0$ | $\to 1$ | Oscillates | Clusters rotate |
| Oscillatory (no cluster) | $\approx 0$ | $\approx 0$ | Oscillates | Non-converging uniform |

## Experimental Setup

### Configuration

All experiments use the following parameter settings:
- **Embedding dimension**: $d = 100$
- **Number of tokens**: $n = 200$ (satisfies $n > d$, consistent with modern LLMs)
- **Initial conditions**: Tokens uniformly distributed on $\mathbb{S}^{99}$ via normalized Gaussian draws
- **ODE solver**: RK45 method with absolute tolerance $10^{-8}$ and relative tolerance $10^{-6}$

### Attention Models Compared

1. **Linear Self-Attention (LSA)**: $h(y) = y$
   - Simple linear attention; theoretical guarantees in 2D
   - Exhibits both antipodal clustering and non-clustering regimes

2. **Unnormalized Softmax Attention (USA)**: $h(y) = \exp(y)$
   - Exponential weighting; different scaling than standard softmax
   - Generally exhibits more aggressive single-point clustering
   - Motivated by recent gradient flow analysis literature

### Key Observation: Model Differences

The experiments reveal a critical qualitative difference:
- **LSA** can exhibit clustering or non-clustering depending on spectral structure of $A$ and $V$
- **USA** tends to produce **single-point consensus** in regimes where LSA forms antipodal clusters
- **Exception**: USA also clusters at single points in regimes where LSA does not cluster at all
- **Conjecture**: If LSA clusters, then USA clusters (converse not necessarily true)

## Experimental Cases and Results

### Case 1: $V = I$, $A$ Arbitrary

**Theoretical Prediction (2D LSA)**: Three regimes determined by spectral properties of $A$:
- $\mathrm{tr}(A) > 0$ → antipodal clustering
- $\mathrm{tr}(A) \leq 0$ and $\det(A) \leq 0$ → antipodal clustering
- $(A+A^\top)/2 \prec 0$ → no clustering

**Experimental Results (d=100, n=200):**

#### Regime 1.1: $\mathrm{tr}(A) > 0$
- **LSA**: Converges to **antipodal clusters** ($\hat{\mathcal{R}}_2(t) \to 1$, $\hat{\mathcal{R}}_1(t) \to 0$)
  - Gram matrix shows 2×2 block structure with balanced cluster sizes
  - Clusters form in approximately 35 time units
- **USA**: Converges to **single point** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - Faster convergence than LSA
  - Qualitatively different: single cluster instead of antipodal

#### Regime 1.2: $\mathrm{tr}(A) \leq 0$ and $\det(A) \leq 0$
- **LSA**: Converges to **antipodal clusters** ($\hat{\mathcal{R}}_2(t) \to 1$, $\hat{\mathcal{R}}_1(t) \to 0$)
  - Qualitatively matches Regime 1.1
  - **Slower convergence**: ~130 time units (vs 35 in Regime 1.1)
  - More balanced cluster distribution
- **USA**: Converges to **single point** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - Similar single-point behavior across regimes
  - More uniform convergence speed than LSA

#### Regime 1.3: $(A+A^\top)/2 \prec 0$ (Partial Synchronization)
- **LSA**: **No clustering** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \approx 0$)
  - Tokens converge without forming clusters
  - Gram matrix shows near-zero off-diagonal entries
- **USA**: Converges to **single point** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - **Clustering occurs despite LSA non-clustering prediction**
  - First evidence of LSA ⊄ USA behavior difference

**Key Insight**: USA exhibits clustering in all three regimes, while LSA distinguishes between clustering and non-clustering based on $A$'s spectrum. This motivates deeper investigation of softmax nonlinearity effects.

---

### Case 2: $A = I$, $V$ Symmetric

**Theoretical Prediction (2D LSA)**: Clustering threshold determined by $\lambda_{\max}(V)$:
- $\lambda_{\max}(V) \geq 0$ → antipodal clustering
- $\lambda_{\max}(V) < 0$ → no clustering

**Experimental Results (d=100, n=200):**

#### Regime 2.1: $\lambda_{\max}(V) > 0$
- **LSA**: Converges to **antipodal clusters** ($\hat{\mathcal{R}}_2(t) \to 1$, $\hat{\mathcal{R}}_1(t) \to 0$)
  - Fast convergence (~40 time units)
  - Gram matrix: balanced 2×2 block structure
- **USA**: Converges to **single point** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - Significantly faster than LSA
  - Different cluster configuration despite same $\lambda_{\max}(V)$ condition

#### Regime 2.2: $\lambda_{\max}(V) = 0$ (Boundary Case)
- **LSA**: Converges to **antipodal clusters** (very slow)
  - $\hat{\mathcal{R}}_2(t) \to 1$ but requires **~150,000 time units**
  - Convergence speed drops **orders of magnitude** from positive eigenvalue regime
  - Gram matrix confirms antipodal structure
- **USA**: Also converges to **antipodal clusters** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - **Only case where USA clusters antipodally** (not single point)
  - Indicates boundary behavior differs from interior regimes

#### Regime 2.3: $\lambda_{\max}(V) < 0$
- **LSA**: **No clustering** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 0$)
  - Coordinate-wise means $m_l(t)$ converge across all dimensions
  - Tokens converge but remain uniformly distributed
- **USA**: **No clustering** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 0$)
  - Both models agree: convergence without cluster formation

**Key Insights**:
1. $\lambda_{\max}(V)$ threshold holds in higher dimensions
2. Boundary case ($\lambda_{\max}(V) = 0$) exhibits critical slowing down
3. At boundary, USA transitions from single-point to antipodal clustering (topological phase shift)

---

### Case 3: $A = I$, $V$ Rotational

**Setup (d=100)**: $V = aI_d + bJ_d$ where $J_d$ is block-diagonal with $\lfloor d/2 \rfloor$ rotation generators

**Theoretical Prediction (2D LSA)**: Clustering with rotation:
- $a > 0$ → antipodal clusters that rotate periodically
- Cluster formation independent of $b$; rotation phase controlled by $b$

**Experimental Results:**

- **LSA**: Forms **antipodal clusters** ($\hat{\mathcal{R}}_2(t) \to 1$, $\hat{\mathcal{R}}_1(t) \to 0$)
  - Gram matrix: 2×2 block structure preserved
  - **Coordinate-wise means $m_l(t)$ oscillate periodically**
  - Clusters rotate: don't reach static equilibrium despite $\hat{\mathcal{R}}_2(t)$ convergence
  - Rotation is **collective** across all coordinate planes

- **USA**: Forms **single cluster** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - Single consensus point but still exhibits **periodic oscillations**
  - All tokens rotate around a common axis

**Key Insight**: Distinguishes **static clustering** from **dynamic rotating clusters**. Gram matrix alone insufficient; requires coordinate-wise mean diagnostic to detect persistent rotation.

---

### Case 4: $A = I$, $V$ Hamiltonian

**Setup (d=100)**:

$$V = \mathrm{blkdiag}\left(\begin{pmatrix}a & b \\ -b & -a\end{pmatrix}, \ldots\right)$$

with $\lfloor d/2 \rfloor$ blocks

**Theoretical Prediction (2D LSA)**: Bifurcation between clustering and oscillation:
- $a > b$ → antipodal clustering
- $a = b$ → critical transition (boundary)
- $a < b$ → persistent periodic oscillations (no clustering)

**Experimental Results:**

#### Regime 4.1: $a > b$
- **LSA**: Converges to **antipodal clusters** ($\hat{\mathcal{R}}_2(t) \to 1$, $\hat{\mathcal{R}}_1(t) \to 0$)
  - Convergence in ~45 time units
  - Unequal cluster distribution (visible in Gram matrix)
- **USA**: Converges to **single point** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 1$)
  - Faster convergence than LSA

#### Regime 4.2: $a = b$ (Critical Boundary)
- **LSA**: Converges to **antipodal clusters** but **extremely slowly**
  - Requires ~200,000 time units (orders of magnitude slower than $a > b$)
  - $\hat{\mathcal{R}}_2(t) \to 1$ but at marginal convergence rate
- **USA**: Converges to **single point** (not antipodal like Case 2 boundary)
  - Indicates different stability structure for Hamiltonian vs. symmetric matrices

#### Regime 4.3: $a < b$ (Oscillatory)
- **LSA**: **Persistent periodic oscillations** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 0$)
  - Gram matrix: no clustering structure
  - Coordinate-wise means $m_l(t)$ oscillate indefinitely
  - Hamiltonian structure produces closed orbits (no convergence)
- **USA**: Also exhibits **persistent oscillations** ($\hat{\mathcal{R}}_1(t), \hat{\mathcal{R}}_2(t) \to 0$)
  - Both models agree on non-convergent behavior
  - Oscillation driven by Hamiltonian structure is model-independent

**Key Insights**:
1. Bifurcation threshold $a = b$ holds across dimensions
2. Hamiltonian structure fundamentally produces closed orbits in oscillatory regime
3. Critical slowing at boundary: one order of magnitude slower than interior
4. USA maintains single-point clustering at boundary (unlike Case 2 where it switches to antipodal)

## Using the Code

### Interactive Visualization

Launch the Streamlit app for real-time parameter exploration:

```bash
streamlit run watanabe.py
```

**Controls:**
- Adjust number of tokens $N$, integration time $T$, number of samples
- Modify inverse temperature $\beta$ (rescaling time; set $\beta = 1$ for standard dynamics)
- Interactively tune matrix entries for $A$ and $V$
- Visualize token trajectories, phase portraits, and clustering metrics

### Running Specific Experiments

To reproduce paper results from different cases:

```bash
# Interactive Streamlit apps with parameter tuning
streamlit run higher_dim.py                  # Run all cases (1-4) in d=100
streamlit run watanabe.py                    # Manual parameter exploration on circle dynamics

# Standalone scripts for validation
python circle_dynamics.py                    # Validate OA on circles (d=1)
python sphere_dynamics.py                    # Validate OA on spheres (d=2)

# Generate all paper figures
streamlit run merged_paper_figures.py
```

**To reproduce a specific experimental regime**, launch the Streamlit app and set:
- **Case 1, Regime 1.1**: $\beta = 1$, adjust $A$ to have $\mathrm{tr}(A) > 0$
- **Case 2, Regime 2.1**: Set $A = I$, construct $V$ with $\lambda_{\max}(V) > 0$
- **Case 3**: Set $A = I$, $V = aI_d + bJ_d$ with $a > 0$
- **Case 4, Regime 4.3**: Set $A = I$, $V$ Hamiltonian with $a < b$

## Key Numerical Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Absolute Tolerance | $10^{-8}$ | ODE integration accuracy |
| Relative Tolerance | $10^{-6}$ | ODE integration accuracy |
| $\rho$ grid minimum | 0.05 | Avoid singularity at $\rho = 0$ |
| Phase portrait grid | $100 \times 100$ | Smooth OA phase portraits |
| $\beta$ scaling | 1.0 (default) | Time rescaling parameter |

## Important Notes

### Singular Behavior at $\rho = 0$

The Ott-Antonsen reduced dynamics contain a division by $\rho$ in the $\dot{\phi}$ equation:

$$\dot{\phi} = \frac{V}{\rho}$$

This is **undefined at $\rho = 0$** (uniform distribution), so phase portraits are computed on $[0.05, 1.0] \times [-\pi, \pi]$.

### Equivalence of Time and $\beta$ Scaling

Since $\beta$ appears as a multiplicative factor in the dynamics, rescaling $\beta$ is equivalent to rescaling time. We therefore fix $\beta = 1$ unless otherwise specified.

### Higher-Dimensional Clustering Direction

For $d > 2$ with Case 1 (Regime 3), the explicit equilibrium formula for $\rho$ is unavailable. Instead, the equilibrium is inferred numerically from the plateau value of $\hat{\mathcal{R}}_2(t)$.

## Reproducibility

All numerical experiments are fully reproducible:

1. **Fixed Random Seeds**: All initialization and matrix construction use seeded RNG
2. **Specified ODE Tolerances**: Absolute tolerance $10^{-8}$, relative tolerance $10^{-6}$ with RK45
3. **Standard Initialization**: Tokens drawn from uniform distribution on sphere via normalized Gaussians
4. **Deterministic Parameter Construction**: Spectral constructions for $A$ and $V$ use prescribed eigenvalues

To verify a result:
```python
# Run the same script twice with same parameter settings
# Results should be numerically identical (within machine precision)
streamlit run higher_dim.py
```

**Important**: The Streamlit apps with random eigenvalue choices will vary slightly due to random basis selection. To reproduce exact figures from the paper, use the parameter specifications provided in each case description above.

## References

For theoretical background and detailed analysis, see:

**Main Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

**Key Sections Implemented**:
- **Ott-Antonsen Reduction**: Classical technique for mean-field dynamics on spheres
- **Case Studies**: Four spectral regimes determining clustering behavior (Section 5 of paper)
- **Higher-Dimensional Generalizations**: Block-diagonal and random-basis constructions
- **LSA vs USA Models**: Comparison of linear and softmax attention mechanisms

**For implementation details**, see [numerics.pdf](numerics.pdf) which contains:
- ODE integration specifications (tolerances, methods)
- Phase portrait computation procedures
- Clustering diagnostic definitions
- Matrix construction guidelines for all cases

## Dependencies

- **NumPy**: Array operations and linear algebra
- **SciPy**: ODE integration (`solve_ivp`), optimization
- **Matplotlib**: Visualization, phase portraits, animations
- **NetworkX**: Graph analysis (if needed for attention patterns)
- **Streamlit**: Interactive web interface
- **SymPy**: Symbolic computation for analytical phase portraits
