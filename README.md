# PDD — Zinc Dendrite Electrodeposition Simulator

Finite-difference solver and Jupyter notebook demo reproducing the **static-state** physics from:

> Jing, Xing, Zhang et al., *Dynamics of zinc dendritic growth in aqueous zinc-based flow batteries: Insights from phase field–Lattice-Boltzmann simulations*, **Chemical Engineering Journal 503 (2025) 158318**.

## Contents

| File | Purpose |
|------|---------|
| `fd_core.py` | NumPy explicit-FD solver: coupled phase field + Nernst-Planck + Poisson |
| `fd_zinc_demo.ipynb` | Runnable notebook demo with parameter sweep and visualizations |
| `Mô phỏng pin kẽm dung môi nước.pdf` | Reference paper |
| `Physics_informed_neural_networks(PINNs)_*.ipynb`, `ExPINN.ipynb` | PINN reference notebooks (separate track) |

## Quick start

```bash
uv run --with numpy --with matplotlib --with jupyter \
    python3 -m jupyter notebook fd_zinc_demo.ipynb
```

Or open `fd_zinc_demo.ipynb` in any Jupyter environment and Run All.

## What the demo shows

- Anisotropic (6-fold) growth of a Zn deposit seeded at the bottom electrode
- Zn²⁺ depletion boundary layer in front of the growing interface — the **"tip effect"**
- Electric potential distribution with the deposit acting as equipotential
- Tunable parameters: exchange-current density proxy `k_dep`, ion diffusion `Ds`, overpotential `E_theta`

## Equations solved

| Field | Equation | Method |
|-------|----------|--------|
| `c_dep` (phase) | Anisotropic Kobayashi Allen-Cahn driven by Butler-Volmer | Explicit FD |
| `c+/c0` (Zn²⁺ conc.) | Nernst-Planck (diffusion + electromigration + reaction sink) | Explicit FD |
| `phi` (potential) | Laplace, deposit pinned as equipotential | Jacobi iteration |

## Simplifications vs the paper

1. Phase-field uses the standard Kobayashi Allen-Cahn form (driven by a bounded `arctan` of the Butler-Volmer expression) instead of the paper's conserved Cahn-Hilliard. Qualitative dendrite physics (tip effect, ion boundary layer, 6-fold anisotropy, effect of `i0`/`Ds`) is identical.
2. **Electrolyte flow (Lattice-Boltzmann, paper Eqs. 5–9) is not included** — this demo is the static state `u = 0`. LBM is phase 2.
3. Dimensional prefactors lumped into nondimensional tunable rates (`k_dep`, `cs_c0`, etc.) with `W0 = 1`, `tau0 = W0² / Ds = 1`.

## Roadmap

1. Add electrolyte flow via Lattice-Boltzmann D2Q9 (paper Eqs. 5–9, Figs. 5–6).
2. Polycrystalline competitive growth (paper Fig. 7).
3. Wrap `run()` in a web interface (Gradio/Flask) with parameter sliders.
4. Optional ML surrogate (FNO/DeepONet) trained on FD ground-truth runs for interactive web speed.
