# PDD — Zinc Dendrite Electrodeposition Simulator

Finite-difference solver, Jupyter demo, and **Gradio web UI** reproducing the physics from:

> Jing, Xing, Zhang et al., *Dynamics of zinc dendritic growth in aqueous zinc-based flow batteries: Insights from phase field–Lattice-Boltzmann simulations*, **Chemical Engineering Journal 503 (2025) 158318**.

## Contents

| File | Purpose |
|------|---------|
| `fd_core.py` | NumPy explicit-FD solver: phase field + Nernst-Planck + Poisson + **LBM flow** + **polycrystalline** |
| `fd_core_numba.py` | `@njit`-compiled version (~1.7× faster) |
| `fd_zinc_demo.ipynb` | Runnable notebook with parameter sweep + visualizations |
| `app.py` | **Gradio web interface** with sliders |
| `Mô phỏng pin kẽm dung môi nước.pdf` | Reference paper |
| `Physics_informed_neural_networks(PINNs)_*.ipynb`, `ExPINN.ipynb` | PINN reference notebooks (separate track) |

## Quick start

### Notebook demo
```bash
uv run --with numpy --with matplotlib --with jupyter \
    python3 -m jupyter notebook fd_zinc_demo.ipynb
```

### Web UI
```bash
uv run --with numpy --with matplotlib --with gradio python3 app.py
# open the printed URL (default http://localhost:7860)
```

### Fast (Numba) backend
```python
from fd_core_numba import run_fast
out = run_fast(steps=6000, u_inlet=0.05)   # ~30s instead of ~50s
```

## Features

- **Phase field** (anisotropic Kobayashi Allen-Cahn driven by bounded Butler-Volmer)
- **Nernst-Planck** for Zn²⁺ (diffusion + electromigration + reaction sink)
- **Poisson / Laplace** for potential (deposit pinned equipotential, Jacobi)
- **Lattice-Boltzmann D2Q9** flow with diffuse-interface drag (paper Eq. 5–9)
- **Convection** of Zn²⁺ via `u·∇c+` coupled from LBM
- **Polycrystalline** multi-seed competitive growth (random `theta_j` per grain, Voronoi orientation field)

## What the demos show

| Phenomenon | Paper | Reproduced in |
|------------|-------|---------------|
| Tip effect (Zn²⁺ depletion boundary layer) | Fig. 2 | static notebook + web UI |
| Effect of exchange current density `i0` | Fig. 3 | notebook `k_dep` sweep |
| Forced-flow dendrite tilt toward inlet | Fig. 5–6 | `u_inlet > 0` in web UI |
| Polycrystalline competition | Fig. 7 | "polycrystalline" checkbox in web UI |

## Equations solved

| Field | Equation | Method |
|-------|----------|--------|
| `c_dep` (phase) | Anisotropic Kobayashi Allen-Cahn driven by Butler-Volmer | Explicit FD |
| `c+/c0` (Zn²⁺) | Nernst-Planck (diffusion + electromigration + advection + sink) | Explicit FD |
| `phi` (potential) | Laplace, deposit pinned | Jacobi iteration |
| `u` (flow) | D2Q9 BGK Lattice-Boltzmann + diffuse-interface drag | LBM |

## Simplifications vs the paper

- Phase-field uses Kobayashi Allen-Cahn (driven by bounded `arctan` of BV) instead of the paper's conserved Cahn-Hilliard. Qualitative dendrite physics is identical.
- Dimensional prefactors lumped into nondimensional tunable rates (`k_dep`, `cs_c0`, etc.) with `W0 = 1`, `tau0 = W0² / Ds = 1`.
- LBM uses bounce-back at top/bottom, equilibrium inlet at right, copy-outflow at left. Body-force-free; flow driven entirely by inlet BC.

## Roadmap

1. ~~LBM flow~~ ✅
2. ~~Polycrystalline competition~~ ✅
3. ~~Gradio web wrapper~~ ✅
4. ~~Numba JIT speedup~~ ✅ (1.7×; further parallel/CUDA possible)
5. Optional ML surrogate (FNO/DeepONet) trained on FD ground-truth.
