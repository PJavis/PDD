"""Gradio web interface for the zinc-dendrite simulator.

Uses the Numba-accelerated backend (fd_core_numba.run_fast) with a fallback
to the pure-NumPy backend (fd_core.run) if Numba isn't installed.

Run:
    uv run --with numpy --with matplotlib --with numba --with gradio python3 app.py
Open the printed local URL in a browser.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    from fd_core_numba import run_fast as _solver
    _BACKEND = "numba"
except Exception:                       # numba missing -> pure numpy
    from fd_core import run as _solver
    _BACKEND = "numpy"


# grid presets: (Nx, Ny) — smaller = faster, larger = more detail
QUALITY = {
    "Fast preview (120x140)": (120, 140),
    "Balanced (160x200)":     (160, 200),
    "High detail (200x280)":  (200, 280),
}


def _render(out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        (out["c"],   "c_dep (metal deposit)", (0, 1),    "viridis"),
        (out["cp"],  "c+ / c0 (Zn2+ ions)",   (0, 1.2),  "plasma"),
        (out["phi"], "phi (V) potential",     (-0.2, 0), "coolwarm"),
    ]
    for a, (f, lab, vm, cm) in zip(ax, panels):
        im = a.imshow(f, origin="lower", aspect="auto", cmap=cm,
                      vmin=vm[0], vmax=vm[1])
        a.set_title(lab); plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    return fig


def _tip_plot(out):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(out["tip_t"], out["tip_len"], "o-")
    ax.set_xlabel("time"); ax.set_ylabel("tallest deposit height")
    ax.set_title("Dendrite growth over time")
    ax.grid(True); plt.tight_layout()
    return fig


def simulate(k_dep, Ds, E_theta, delta, u_inlet, steps, quality,
             multi_seed, n_seeds):
    Nx, Ny = QUALITY[quality]
    if multi_seed:
        rng = np.random.default_rng(1)
        xs = np.linspace(0.15 * Nx, 0.85 * Nx, int(n_seeds))
        seeds = [(x, 0, 6.0, np.deg2rad(rng.uniform(0, 60))) for x in xs]
    else:
        seeds = None

    out = _solver(
        Nx=Nx, Ny=Ny, steps=int(steps),
        k_dep=float(k_dep), Ds=float(Ds), E_theta=float(E_theta),
        delta=float(delta), u_inlet=float(u_inlet),
        seeds=seeds,
        record_every=max(1, int(steps) // 5),
        verbose=False,
    )
    fields_fig = _render(out)
    tip_fig = _tip_plot(out)

    # Damkohler-style ratio: reaction rate / transport rate.
    # High Da -> reaction-limited (ions consumed before reaching tip,
    #            ramified/branched growth). Low Da -> compact growth.
    Da = float(k_dep) / (10.0 * float(Ds))
    regime = ("reaction-limited (ramified/branched)" if Da > 1.5
              else "transport-limited (compact)" if Da < 0.6
              else "mixed")
    summary = (
        f"**Backend:** {_BACKEND}  |  **grid:** {Nx}x{Ny}  \n"
        f"**Damkohler Da = k_dep / (10*Ds) = {Da:.2f}** -> {regime}  \n"
        f"final height: {out['tip_len'][-1]:.0f}  |  "
        f"max c_dep: {out['c'].max():.3f}  |  "
        f"min c+: {out['cp'].min():.3f} (lower = stronger ion depletion)"
    )
    return fields_fig, tip_fig, summary


with gr.Blocks(title="Zinc Dendrite Simulator") as demo:
    gr.Markdown(
        "# Zinc Dendrite Electrodeposition Simulator\n"
        "Reproduces qualitative physics of *Jing et al., Chem. Eng. J. 503 "
        "(2025) 158318* — phase field + Nernst-Planck + Poisson, optional LBM "
        "flow.\n\n"
        "**Tip:** the morphology is governed by the **Damkohler number** "
        "`Da ~ k_dep / Ds` (reaction vs ion transport). Raise `k_dep` or lower "
        "`Ds` for tall, ramified growth; do the opposite for compact deposits."
    )
    with gr.Row():
        with gr.Column(scale=1):
            k_dep   = gr.Slider(1.0, 40.0, value=16.0, step=0.5,
                                label="k_dep  (reaction rate ~ exchange current i0)")
            Ds      = gr.Slider(0.2, 5.0,  value=1.0,  step=0.1,
                                label="Ds  (Zn2+ diffusion / transport speed)")
            E_theta = gr.Slider(-0.5, -0.1, value=-0.3, step=0.02,
                                label="E_theta (V)  (more negative = stronger push)")
            delta   = gr.Slider(0.0, 0.35, value=0.1,  step=0.01,
                                label="delta  (anisotropy / branchiness)")
            u_inlet = gr.Slider(0.0, 0.1,  value=0.0,  step=0.005,
                                label="u_inlet  (electrolyte flow; 0 = static)")
            steps   = gr.Slider(1000, 12000, value=6000, step=500,
                                label="steps  (longer = taller dendrite, slower)")
            quality = gr.Dropdown(list(QUALITY.keys()),
                                  value="Balanced (160x200)", label="Quality / grid")
            multi_seed = gr.Checkbox(value=False, label="Polycrystalline (many seeds)")
            n_seeds = gr.Slider(3, 16, value=8, step=1, label="n_seeds (if polycrystalline)")
            run_btn = gr.Button("Run simulation", variant="primary")
        with gr.Column(scale=2):
            fields = gr.Plot(label="Final fields")
            tip = gr.Plot(label="Growth vs time")
            summary = gr.Markdown()
    run_btn.click(
        simulate,
        inputs=[k_dep, Ds, E_theta, delta, u_inlet, steps, quality,
                multi_seed, n_seeds],
        outputs=[fields, tip, summary],
    )


if __name__ == "__main__":
    print(f"Backend: {_BACKEND}")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
