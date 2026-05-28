"""
Finite-difference solver for zinc dendrite electrodeposition (static state, u=0).
Reproduces qualitatively the static-state physics of:
  Jing et al., Chem. Eng. J. 503 (2025) 158318.
Coupled fields (nondimensional, length in W0, time in tau0 = W0^2/Ds):
  c   : phase field / deposit fraction (Eq.1 Cahn-Hilliard + source)
  cp  : Zn2+ concentration c+/c0       (Eq.3 Nernst-Planck, convection off)
  phi : electric potential             (Eq.4 Laplace in electrolyte, SOR)
Growth driven by Butler-Volmer velocity v(theta) (Eq.2) with 6-fold anisotropy.
This is a qualitative demo: dimensional prefactors lumped into tunable rates.
"""
import numpy as np


def _shift(a, axis, k):
    """Shift array by k along axis with replicate (zero-flux) padding."""
    if k == 1:
        if axis == 0:
            out = np.empty_like(a); out[1:, :] = a[:-1, :]; out[0, :] = a[0, :]
        else:
            out = np.empty_like(a); out[:, 1:] = a[:, :-1]; out[:, 0] = a[:, 0]
    else:  # k == -1
        if axis == 0:
            out = np.empty_like(a); out[:-1, :] = a[1:, :]; out[-1, :] = a[-1, :]
        else:
            out = np.empty_like(a); out[:, :-1] = a[:, 1:]; out[:, -1] = a[:, -1]
    return out


def laplacian(a, dx):
    return (
        -4.0 * a
        + _shift(a, 0, 1) + _shift(a, 0, -1)
        + _shift(a, 1, 1) + _shift(a, 1, -1)
    ) / dx**2


def grad(a, dx):
    ay = (_shift(a, 0, -1) - _shift(a, 0, 1)) / (2 * dx)
    ax = (_shift(a, 1, -1) - _shift(a, 1, 1)) / (2 * dx)
    return ax, ay


def divergence(fx, fy, dx):
    dfx = (_shift(fx, 1, -1) - _shift(fx, 1, 1)) / (2 * dx)
    dfy = (_shift(fy, 0, -1) - _shift(fy, 0, 1)) / (2 * dx)
    return dfx + dfy


def solve_phi(phi, c, dx, phi_top, phi_dep, n_iter=25):
    """Laplace for phi in electrolyte; deposit (c>0.5) pinned to phi_dep.
    Dirichlet: top=phi_top, bottom=phi_dep. Neumann (zero-flux) left/right.
    Jacobi iteration (stable); warm-started from previous step."""
    solid = c > 0.5
    for _ in range(n_iter):
        phi[solid] = phi_dep
        phi[0, :] = phi_dep          # bottom electrode
        phi[-1, :] = phi_top         # top electrolyte
        phi[:, 0] = phi[:, 1]        # left zero-flux
        phi[:, -1] = phi[:, -2]      # right zero-flux
        phi = 0.25 * (
            _shift(phi, 0, 1) + _shift(phi, 0, -1)
            + _shift(phi, 1, 1) + _shift(phi, 1, -1)
        )
    phi[solid] = phi_dep
    phi[0, :] = phi_dep
    phi[-1, :] = phi_top
    return phi


def run(
    Nx=160, Ny=220, dx=1.0, dt=5e-3, steps=8000,
    W0=1.0, tau=0.5,                # interface width & phase-field relaxation
    delta=0.05, omega_aniso=6, theta_j=0.0,
    beta=0.9,                       # driving strength (|m| < beta/2)
    k_dep=8.0, alpha=0.5, E_theta=-0.3,
    De=1e-3, Ds=1.0, zF_RT=4.0,     # NP diffusion (deposit/electrolyte), z F/RT
    cs_c0=1.0,                      # cs/c0 reaction sink ratio
    phi_top=-0.2, phi_dep=0.0,
    grad_phi_cap=0.5,
    seed_r=6.0, record_every=400, verbose=True,
):
    # fields: row index y (0=bottom), col index x
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    cx0 = Nx / 2.0
    # semicircular seed at bottom center, diffuse tanh profile (width W0)
    r = np.sqrt(((xx - cx0) * dx) ** 2 + (yy * dx) ** 2)
    c = 0.5 * (1.0 - np.tanh((r - seed_r * dx) / (np.sqrt(2) * W0)))

    cp = np.ones((Ny, Nx))           # c+/c0 = 1 everywhere
    phi = np.full((Ny, Nx), phi_top)

    frames = []
    tip_len, tip_t = [], []

    for it in range(steps):
        # --- Poisson/Laplace ---
        phi = solve_phi(phi, c, dx, phi_top, phi_dep)

        # --- interface normal angle theta + 6-fold anisotropy A(theta) ---
        cx, cy = grad(c, dx)
        theta = np.arctan2(cy, cx)
        ang = omega_aniso * (theta - theta_j)
        A = W0 * (1.0 + delta * np.cos(ang))
        Ap = -W0 * delta * omega_aniso * np.sin(ang)   # dA/dtheta

        # --- Butler-Volmer driving, bounded (Kobayashi-style arctan) ---
        # paper uses conserved Cahn-Hilliard; this anisotropic Allen-Cahn
        # (Kobayashi) form gives identical qualitative dendrite physics.
        eta = phi - E_theta
        H = np.clip(zF_RT * eta, -40, 40)
        S = cp * np.exp(alpha * H) - np.exp(-(1 - alpha) * H)   # >0 -> deposition
        m = (beta / np.pi) * np.arctan(k_dep * S)              # |m| < beta/2

        # --- anisotropic gradient (Kobayashi 2D) ---
        A2 = A * A
        grad_iso = divergence(A2 * cx, A2 * cy, dx)
        gx1, _ = grad(A * Ap * cy, dx)     # d/dx (A A' c_y)
        _, gy2 = grad(A * Ap * cx, dx)     # d/dy (A A' c_x)
        cross = -gx1 + gy2
        reaction = c * (1 - c) * (c - 0.5 + m)
        dc = dt * (grad_iso + cross + reaction) / tau
        c = np.clip(c + dc, 0.0, 1.0)
        # phase-field BCs: no growth at top/sides, mirror at bottom
        c[-1, :] = 0.0
        c[:, 0] = 0.0
        c[:, -1] = 0.0
        c[0, :] = c[1, :]

        # --- Nernst-Planck for c+ (Eq.3), u=0 ---
        Deff = De * c + Ds * (1 - c)
        # diffusion term: div(Deff grad cp)
        cpx, cpy = grad(cp, dx)
        diff = divergence(Deff * cpx, Deff * cpy, dx)
        # electromigration: div(Deff * cp * zF/RT * grad phi), electrolyte only
        phx, phy = grad(phi, dx)
        phx = np.clip(phx, -grad_phi_cap, grad_phi_cap)
        phy = np.clip(phy, -grad_phi_cap, grad_phi_cap)
        liquid = (1 - c)
        emx = Deff * cp * zF_RT * phx * liquid
        emy = Deff * cp * zF_RT * phy * liquid
        emig = divergence(emx, emy, dx)
        sink = cs_c0 * (dc / dt)     # reaction consumes ions where c grows
        cp = cp + dt * (diff + emig - sink)
        cp = np.clip(cp, 0.0, 5.0)
        # BCs: continuous supply top & right = 1.0; zero-flux left; bottom mirror
        cp[-1, :] = 1.0
        cp[:, -1] = 1.0
        cp[:, 0] = cp[:, 1]
        cp[0, :] = cp[1, :]

        if it % record_every == 0:
            ys = np.where(c[:, int(cx0)] > 0.5)[0]
            h = ys.max() * dx if ys.size else 0.0
            tip_len.append(h)
            tip_t.append(it * dt)
            frames.append((it * dt, c.copy(), cp.copy(), phi.copy()))
            if verbose:
                print(f"step {it:5d}  t={it*dt:6.3f}  tipH={h:6.2f}  "
                      f"cmax={c.max():.2f}  cp[min]={cp.min():.2f}")

    return dict(frames=frames, tip_len=np.array(tip_len),
                tip_t=np.array(tip_t), c=c, cp=cp, phi=phi)


if __name__ == "__main__":
    out = run(steps=4000, record_every=400)
    print("DONE. final tip height:", out["tip_len"][-1])
