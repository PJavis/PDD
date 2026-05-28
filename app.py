"""Gradio web interface for the zinc-dendrite FD solver.

Run:
    uv run --with numpy --with matplotlib --with gradio python3 app.py
Open the printed local URL in a browser.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from fd_core import run


def _render(out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [
        (out["c"],   "c_dep",   (0, 1),    "viridis"),
        (out["cp"],  "c+ / c0", (0, 1.2),  "plasma"),
        (out["phi"], "phi (V)", (-0.2, 0), "coolwarm"),
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
    ax.set_xlabel("t"); ax.set_ylabel("tip height")
    ax.set_title("Dendrite tip growth")
    ax.grid(True); plt.tight_layout()
    return fig


def simulate(k_dep, Ds, E_theta, u_inlet, steps, multi_seed, n_seeds):
    if multi_seed:
        rng = np.random.default_rng(1)
        xs = np.linspace(20, 140, int(n_seeds))
        seeds = [(x, 0, 6.0, np.deg2rad(rng.uniform(0, 60))) for x in xs]
        Ny = 160
    else:
        seeds = None
        Ny = 220
    out = run(
        Nx=160, Ny=Ny, steps=int(steps),
        k_dep=float(k_dep), Ds=float(Ds), E_theta=float(E_theta),
        u_inlet=float(u_inlet),
        seeds=seeds,
        record_every=max(1, int(steps) // 4),
        verbose=False,
    )
    fields_fig = _render(out)
    tip_fig = _tip_plot(out)
    summary = (
        f"final tip height: {out['tip_len'][-1]:.1f}  |  "
        f"max c_dep: {out['c'].max():.3f}  |  "
        f"min c+: {out['cp'].min():.3f}"
    )
    return fields_fig, tip_fig, summary


with gr.Blocks(title="Zinc Dendrite Simulator") as demo:
    gr.Markdown("# Zinc Dendrite Electrodeposition Simulator\n"
                "Reproduces qualitative physics of *Jing et al., "
                "Chem. Eng. J. 503 (2025) 158318* — phase field + Nernst-Planck "
                "+ Poisson, optional LBM flow.")
    with gr.Row():
        with gr.Column(scale=1):
            k_dep   = gr.Slider(2.0, 32.0, value=8.0,   step=0.5,
                                label="k_dep (exchange-current proxy ~ i0)")
            Ds      = gr.Slider(0.2, 3.0,  value=1.0,   step=0.1,
                                label="Ds (ion diffusion)")
            E_theta = gr.Slider(-0.5, -0.1, value=-0.3, step=0.02,
                                label="E_theta (V, std half-cell potential)")
            u_inlet = gr.Slider(0.0, 0.1,  value=0.0,   step=0.005,
                                label="u_inlet (LBM flow; 0 = static state)")
            steps   = gr.Slider(1000, 8000, value=4000, step=500,
                                label="steps (more = slower, taller dendrite)")
            multi_seed = gr.Checkbox(value=False, label="Polycrystalline (multi-seed)")
            n_seeds = gr.Slider(3, 12, value=6, step=1, label="n_seeds (if multi)")
            run_btn = gr.Button("Run simulation", variant="primary")
        with gr.Column(scale=2):
            fields = gr.Plot(label="Final fields")
            tip = gr.Plot(label="Tip growth vs time")
            summary = gr.Markdown()
    run_btn.click(
        simulate,
        inputs=[k_dep, Ds, E_theta, u_inlet, steps, multi_seed, n_seeds],
        outputs=[fields, tip, summary],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
