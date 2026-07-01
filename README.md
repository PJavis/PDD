# PDD — Zinc Dendrite Electrodeposition Simulator

Finite-difference solver, Jupyter demo, and **Gradio web UI** reproducing the physics from:

> Jing, Xing, Zhang et al., *Dynamics of zinc dendritic growth in aqueous zinc-based flow batteries: Insights from phase field–Lattice-Boltzmann simulations*, **Chemical Engineering Journal 503 (2025) 158318**.

> 🟢 **New here / not a programmer?** Read **[GETTING_STARTED.md](GETTING_STARTED.md)** — a click-by-click guide that installs everything from scratch and runs the web demo. No coding experience needed.

> 🇻🇳 **Tiếng Việt:** giải thích *chương trình giải phương trình gì, tại sao, và bằng cách nào* — xem **[TAI_LIEU_VI.md](TAI_LIEU_VI.md)**.

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
# include --with numba so the UI auto-uses the fast backend
uv run --with numpy --with matplotlib --with numba --with gradio python3 app.py
# open the printed URL (default http://localhost:7860)
```
The UI prints `Backend: numba` (or `numpy` if Numba is absent). Sliders:
`k_dep`, `Ds`, `E_theta`, `delta`, `u_inlet`, `steps` (up to **200 000** for
very tall/fully-developed dendrites — but mind the run-time estimate), a
grid-quality preset, and a polycrystalline toggle. The result panel shows the
Damköhler number `Da ~ k_dep/Ds` and its regime (compact / mixed / ramified).

> **Grid size** is chosen via the **Quality / grid** dropdown (Fast preview
> `120×140` → Very high detail `280×400`), or set **`Custom`** to type your own
> `Nx` (width, 64–400) and `Ny` (height, 64–512) on the two sliders. Presets
> keep beginners from accidentally picking a browser-freezing grid; Custom gives
> full control when you need it.

Slider ranges (widened for stronger effects, all within the explicit-FD
stability limit): `k_dep` 0.5–80, `Ds` 0.1–10, `E_theta` −0.8…−0.02,
`delta` 0.0–0.6, `u_inlet` 0.0–2.0, `steps` 1000–200000, `n_seeds` 2–24.

> **`u_inlet` is a 0–2 flow-strength dial.** Lattice-Boltzmann is only stable
> at low Mach number, so the slider is mapped internally onto a safe lattice
> velocity (`2.0 → 0.2` lattice units). Higher = visibly stronger flow with no
> NaN blow-ups; tune `U_INLET_MAX_LATTICE` in `app.py` to change the ceiling.

### Fast (Numba) backend
```python
from fd_core_numba import run_fast
out = run_fast(steps=6000, u_inlet=0.05)   # ~2.5x faster than naive NumPy loop
```

## Features

- **Phase field** (anisotropic Kobayashi Allen-Cahn driven by Butler-Volmer)
- **Nernst-Planck** for Zn²⁺ (diffusion + electromigration + reaction sink)
- **Poisson / Laplace** for potential (deposit pinned equipotential, Jacobi)
- **Lattice-Boltzmann D2Q9** flow with diffuse-interface drag (paper Eq. 5–9)
- **Convection** of Zn²⁺ via `u·∇c+` coupled from LBM
- **Polycrystalline** multi-seed competitive growth (random `theta_j` per grain, Voronoi orientation field)
- **Damköhler-controlled morphology**: `Da ~ k_dep / Ds` selects compact vs ramified growth (paper Fig. 3 physics)
- **Numba backend** + Poisson-every-K-steps → ~2.5× faster than the naïve loop
- **Web UI** with grid-size quality presets (fast preview → high detail) and a
  **live run-time estimate** next to the Run button

## Performance / "why is it slow?"

Run time scales **linearly with `Nx · Ny · steps`** — roughly `1.2e-7 s` per
cell per step on the Numba backend. So the cost of the built-in configs is:

| Config | Approx. time |
|--------|-------------|
| Balanced 160×200, 6000 steps | ~12 s |
| Very-high-detail 280×400, 6000 steps | ~85 s |
| 280×400, 24000 steps | ~5 min |
| 160×200, 200000 steps (max) | ~13 min |
| 280×400, 200000 steps (max) | ~45 min |
| Custom 400×512 + flow, 200000 steps (max) | ~2 hr |

Enabling flow (`u_inlet > 0`) adds ~60% (the LBM step). The **biggest lever is
just picking a smaller grid / fewer steps while exploring**, then scaling up for
the final picture — the UI shows the estimate so you know before clicking Run.

> **Why not multi-threaded?** The kernels are single-threaded `@njit` on
> purpose. Per-pass `parallel=True`/`prange` was implemented and benchmarked and
> ran **~4–6× slower** here: the solver is many small stencil passes, and the
> parallel transform loses SIMD vectorization and pays fork-join overhead on
> every pass, every step. Real speedups would need a *different* structure
> (e.g. one fused kernel, or a GPU port) — see roadmap.

### Responsive driving (why sliders now visibly change the result)

The phase-field driving uses `m = m_max · tanh(k_dep·S / k_ref)`. The earlier
`arctan` form **saturated** — once `k_dep·S` was large, doubling `k_dep` barely
moved `m`, so wildly different inputs produced near-identical figures. The `tanh`
form with the `k_ref` gain keeps the slider operating in its sensitive band, so
`k_dep` (reaction rate) and `Ds` (transport) both change the morphology, exactly
as the Damköhler number predicts.

## What the demos show

| Phenomenon | Paper | Reproduced in |
|------------|-------|---------------|
| Tip effect (Zn²⁺ depletion boundary layer) | Fig. 2 | static notebook + web UI |
| Effect of exchange current density `i0` | Fig. 3 | notebook `k_dep` sweep |
| Damköhler ratio `Da~k_dep/Ds` → compact vs ramified | Fig. 3 + text | notebook Damköhler sweep + web UI readout |
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

- Phase-field uses Kobayashi Allen-Cahn (driven by `m = m_max·tanh(k_dep·S/k_ref)` of the Butler-Volmer expression `S`) instead of the paper's conserved Cahn-Hilliard. Qualitative dendrite physics is identical. (`tanh` chosen over `arctan` to avoid driving saturation — see "Responsive driving" above.)
- Dimensional prefactors lumped into nondimensional tunable rates (`k_dep`, `cs_c0`, etc.) with `W0 = 1`, `tau0 = W0² / Ds = 1`.
- LBM uses bounce-back at top/bottom, equilibrium inlet at right, copy-outflow at left. Body-force-free; flow driven entirely by inlet BC.

## Roadmap

1. ~~LBM flow~~ ✅
2. ~~Polycrystalline competition~~ ✅
3. ~~Gradio web wrapper~~ ✅
4. ~~Numba JIT speedup~~ ✅ (1.7×; ~2.5× combined with `phi_every`)
5. ~~Responsive driving + wide parameter range + Damköhler readout~~ ✅
6. Optional ML surrogate (FNO/DeepONet) trained on FD ground-truth.
7. Further speed: per-pass `parallel`/`prange` was tried and is **~4–6× slower**
   (see Performance). Real gains need a single fused kernel or a CUDA/GPU port.
