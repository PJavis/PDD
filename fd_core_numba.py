"""Numba-JIT accelerated version of fd_core.run().

Expected speedup ~10-30x over the pure-NumPy version for the same grid size.
Compile cost is paid once on first call (~5-10s).

Usage:
    from fd_core_numba import run_fast
    out = run_fast(...)   # same signature & return as fd_core.run()
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _solve_phi_jit(phi, c, phi_top, phi_dep, n_iter):
    Ny, Nx = phi.shape
    for _ in range(n_iter):
        # BCs
        for j in range(Nx):
            phi[0, j] = phi_dep
            phi[Ny - 1, j] = phi_top
        for i in range(Ny):
            phi[i, 0] = phi[i, 1]
            phi[i, Nx - 1] = phi[i, Nx - 2]
        # pin solid to phi_dep
        for i in range(Ny):
            for j in range(Nx):
                if c[i, j] > 0.5:
                    phi[i, j] = phi_dep
        # Jacobi update with replicate-edge boundaries
        new_phi = np.empty_like(phi)
        for i in range(Ny):
            ip = i + 1 if i < Ny - 1 else i
            im = i - 1 if i > 0 else i
            for j in range(Nx):
                jp = j + 1 if j < Nx - 1 else j
                jm = j - 1 if j > 0 else j
                new_phi[i, j] = 0.25 * (phi[ip, j] + phi[im, j]
                                        + phi[i, jp] + phi[i, jm])
        for i in range(Ny):
            for j in range(Nx):
                phi[i, j] = new_phi[i, j]
    # final pin
    for i in range(Ny):
        for j in range(Nx):
            if c[i, j] > 0.5:
                phi[i, j] = phi_dep
        phi[i, 0] = phi[i, 1]
        phi[i, Nx - 1] = phi[i, Nx - 2]
    for j in range(Nx):
        phi[0, j] = phi_dep
        phi[Ny - 1, j] = phi_top
    return phi


# D2Q9 lattice constants
LBM_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
LBM_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
LBM_W = np.array([4. / 9.] + [1. / 9.] * 4 + [1. / 36.] * 4)


@njit(cache=True, fastmath=True)
def _lbm_step_jit(f, c, nu, W0, u_inlet, h_drag=2.757):
    Ny, Nx = c.shape
    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    rho = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            s = 0.0; sx = 0.0; sy = 0.0
            for k in range(9):
                s += f[k, i, j]
                sx += f[k, i, j] * LBM_EX[k]
                sy += f[k, i, j] * LBM_EY[k]
            rho[i, j] = s
            ux[i, j] = sx / s
            uy[i, j] = sy / s
    # drag + half-step force
    inv_tauf = 1.0 / (3.0 * nu + 0.5)
    factor = 1.0 - inv_tauf * 0.5
    for i in range(Ny):
        for j in range(Nx):
            drag = 2.0 * nu * h_drag * (1.0 - c[i, j]) * c[i, j] * c[i, j] / (W0 * W0)
            Fx = -rho[i, j] * drag * ux[i, j]
            Fy = -rho[i, j] * drag * uy[i, j]
            ux2 = ux[i, j] + Fx / (2.0 * rho[i, j])
            uy2 = uy[i, j] + Fy / (2.0 * rho[i, j])
            uu = ux2 * ux2 + uy2 * uy2
            for k in range(9):
                ex = LBM_EX[k]; ey = LBM_EY[k]
                eu = ex * ux2 + ey * uy2
                feq = LBM_W[k] * rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
                Gi = LBM_W[k] * factor * (
                    3.0 * ((ex - ux2) * Fx + (ey - uy2) * Fy)
                    + 9.0 * eu * (ex * Fx + ey * Fy)
                )
                f[k, i, j] = f[k, i, j] - (f[k, i, j] - feq) * inv_tauf + Gi
            ux[i, j] = ux2; uy[i, j] = uy2
    # streaming with periodic roll
    new_f = np.empty_like(f)
    for k in range(9):
        ex = LBM_EX[k]; ey = LBM_EY[k]
        for i in range(Ny):
            si = (i - ey) % Ny
            for j in range(Nx):
                sj = (j - ex) % Nx
                new_f[k, i, j] = f[k, si, sj]
    # bounce-back at top/bottom
    OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
    for k in range(9):
        if LBM_EY[k] > 0:
            for j in range(Nx):
                new_f[k, 0, j] = new_f[OPP[k], 0, j]
        elif LBM_EY[k] < 0:
            for j in range(Nx):
                new_f[k, Ny - 1, j] = new_f[OPP[k], Ny - 1, j]
    # right inlet (equilibrium with prescribed u)
    for i in range(Ny):
        uu_in = u_inlet * u_inlet
        for k in range(9):
            ex = LBM_EX[k]; ey = LBM_EY[k]
            eu = ex * (-u_inlet)
            new_f[k, i, Nx - 1] = LBM_W[k] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu_in)
    # left outflow: copy
    for k in range(9):
        for i in range(Ny):
            new_f[k, i, 0] = new_f[k, i, 1]
    return new_f, ux, uy


@njit(cache=True, fastmath=True)
def _phase_np_step_jit(c, cp, phi, ux, uy, grain_theta,
                      dx, dt, W0, tau, delta, omega_aniso, beta, k_dep, alpha,
                      E_theta, De, Ds, zF_RT, cs_c0, grad_phi_cap, use_lbm):
    Ny, Nx = c.shape
    # spatial derivatives (replicate-edge BC)
    cx = np.empty((Ny, Nx)); cy = np.empty((Ny, Nx))
    for i in range(Ny):
        ip = i + 1 if i < Ny - 1 else i
        im = i - 1 if i > 0 else i
        for j in range(Nx):
            jp = j + 1 if j < Nx - 1 else j
            jm = j - 1 if j > 0 else j
            cx[i, j] = (c[i, jp] - c[i, jm]) / (2.0 * dx)
            cy[i, j] = (c[ip, j] - c[im, j]) / (2.0 * dx)

    # anisotropy A(theta)
    A = np.empty((Ny, Nx)); Ap = np.empty((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            theta = np.arctan2(cy[i, j], cx[i, j])
            ang = omega_aniso * (theta - grain_theta[i, j])
            A[i, j] = W0 * (1.0 + delta * np.cos(ang))
            Ap[i, j] = -W0 * delta * omega_aniso * np.sin(ang)

    # Butler-Volmer driving m
    m = np.empty((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            eta = phi[i, j] - E_theta
            H = zF_RT * eta
            if H > 40.0: H = 40.0
            if H < -40.0: H = -40.0
            S = cp[i, j] * np.exp(alpha * H) - np.exp(-(1.0 - alpha) * H)
            m[i, j] = (beta / np.pi) * np.arctan(k_dep * S)

    # phase-field grad terms (anisotropic Kobayashi)
    A2cx = np.empty((Ny, Nx)); A2cy = np.empty((Ny, Nx))
    AApcy = np.empty((Ny, Nx)); AApcx = np.empty((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            A2cx[i, j] = A[i, j] * A[i, j] * cx[i, j]
            A2cy[i, j] = A[i, j] * A[i, j] * cy[i, j]
            AApcy[i, j] = A[i, j] * Ap[i, j] * cy[i, j]
            AApcx[i, j] = A[i, j] * Ap[i, j] * cx[i, j]

    # divergence + cross terms (replicate-edge)
    dc = np.empty((Ny, Nx))
    for i in range(Ny):
        ip = i + 1 if i < Ny - 1 else i
        im = i - 1 if i > 0 else i
        for j in range(Nx):
            jp = j + 1 if j < Nx - 1 else j
            jm = j - 1 if j > 0 else j
            grad_iso = ((A2cx[i, jp] - A2cx[i, jm]) / (2 * dx)
                        + (A2cy[ip, j] - A2cy[im, j]) / (2 * dx))
            gx1 = (AApcy[i, jp] - AApcy[i, jm]) / (2 * dx)
            gy2 = (AApcx[ip, j] - AApcx[im, j]) / (2 * dx)
            cross = -gx1 + gy2
            reaction = c[i, j] * (1.0 - c[i, j]) * (c[i, j] - 0.5 + m[i, j])
            dc[i, j] = dt * (grad_iso + cross + reaction) / tau

    new_c = np.empty((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            v = c[i, j] + dc[i, j]
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            new_c[i, j] = v
    # phase-field BCs
    for j in range(Nx):
        new_c[Ny - 1, j] = 0.0
    for i in range(Ny):
        new_c[i, 0] = 0.0
        new_c[i, Nx - 1] = 0.0
    for j in range(Nx):
        new_c[0, j] = new_c[1, j]

    # Nernst-Planck for cp (diffusion + electromigration + sink + advection)
    new_cp = np.empty((Ny, Nx))
    for i in range(Ny):
        ip = i + 1 if i < Ny - 1 else i
        im = i - 1 if i > 0 else i
        for j in range(Nx):
            jp = j + 1 if j < Nx - 1 else j
            jm = j - 1 if j > 0 else j
            Deff_c = De * c[i, j] + Ds * (1.0 - c[i, j])
            Deff_ip = De * c[ip, j] + Ds * (1.0 - c[ip, j])
            Deff_im = De * c[im, j] + Ds * (1.0 - c[im, j])
            Deff_jp = De * c[i, jp] + Ds * (1.0 - c[i, jp])
            Deff_jm = De * c[i, jm] + Ds * (1.0 - c[i, jm])
            cpx_c = (cp[i, jp] - cp[i, jm]) / (2.0 * dx)
            cpy_c = (cp[ip, j] - cp[im, j]) / (2.0 * dx)
            # div(D grad cp) using central
            diff = ((Deff_jp * (cp[i, jp] - cp[i, j])
                     - Deff_jm * (cp[i, j] - cp[i, jm])) / (dx * dx)
                    + (Deff_ip * (cp[ip, j] - cp[i, j])
                       - Deff_im * (cp[i, j] - cp[im, j])) / (dx * dx))
            # electromigration grad phi capped
            phx = (phi[i, jp] - phi[i, jm]) / (2.0 * dx)
            phy = (phi[ip, j] - phi[im, j]) / (2.0 * dx)
            if phx > grad_phi_cap: phx = grad_phi_cap
            if phx < -grad_phi_cap: phx = -grad_phi_cap
            if phy > grad_phi_cap: phy = grad_phi_cap
            if phy < -grad_phi_cap: phy = -grad_phi_cap
            liq = 1.0 - c[i, j]
            emx = Deff_c * cp[i, j] * zF_RT * phx * liq
            emy = Deff_c * cp[i, j] * zF_RT * phy * liq
            # div via finite diff of emx,emy (rough but fine for demo)
            # use neighbor flux differences
            emx_jp = Deff_jp * cp[i, jp] * zF_RT * ((phi[i, jp] - phi[i, j]) / dx) * (1 - c[i, jp])
            emx_jm = Deff_jm * cp[i, jm] * zF_RT * ((phi[i, j] - phi[i, jm]) / dx) * (1 - c[i, jm])
            emy_ip = Deff_ip * cp[ip, j] * zF_RT * ((phi[ip, j] - phi[i, j]) / dx) * (1 - c[ip, j])
            emy_im = Deff_im * cp[im, j] * zF_RT * ((phi[i, j] - phi[im, j]) / dx) * (1 - c[im, j])
            emig = ((emx_jp - emx_jm) + (emy_ip - emy_im)) / (2.0 * dx)
            sink = cs_c0 * dc[i, j] / dt
            val = cp[i, j] + dt * (diff + emig - sink)
            if use_lbm:
                val = val - (ux[i, j] * cpx_c + uy[i, j] * cpy_c) * liq
            if val < 0.0: val = 0.0
            if val > 5.0: val = 5.0
            new_cp[i, j] = val
    # BCs
    for j in range(Nx):
        new_cp[Ny - 1, j] = 1.0
        new_cp[0, j] = new_cp[1, j]
    for i in range(Ny):
        new_cp[i, Nx - 1] = 1.0
        new_cp[i, 0] = new_cp[i, 1]

    return new_c, new_cp, dc


def _lbm_init(Ny, Nx, u_inlet):
    rho = np.ones((Ny, Nx))
    ux = -u_inlet * np.ones((Ny, Nx)); uy = np.zeros((Ny, Nx))
    f = np.empty((9, Ny, Nx))
    uu = ux * ux + uy * uy
    for k in range(9):
        ex, ey = LBM_EX[k], LBM_EY[k]
        eu = ex * ux + ey * uy
        f[k] = LBM_W[k] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uu)
    return f


def run_fast(
    Nx=160, Ny=220, dx=1.0, dt=5e-3, steps=8000,
    W0=1.0, tau=0.5,
    delta=0.05, omega_aniso=6, theta_j=0.0,
    beta=0.9,
    k_dep=8.0, alpha=0.5, E_theta=-0.3,
    De=1e-3, Ds=1.0, zF_RT=4.0,
    cs_c0=1.0,
    phi_top=-0.2, phi_dep=0.0,
    grad_phi_cap=0.5,
    seed_r=6.0, record_every=400, verbose=True,
    u_inlet=0.0, nu_lbm=0.1,
    seeds=None, n_phi_iter=25,
):
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    cx0 = Nx / 2.0

    if seeds is None:
        r = np.sqrt(((xx - cx0) * dx) ** 2 + (yy * dx) ** 2)
        c = 0.5 * (1.0 - np.tanh((r - seed_r * dx) / (np.sqrt(2) * W0)))
        grain_theta = np.full((Ny, Nx), theta_j)
    else:
        c = np.zeros((Ny, Nx)); grain_theta = np.zeros((Ny, Nx))
        dist_min = np.full((Ny, Nx), np.inf)
        for (sx, sy, sr, st) in seeds:
            r = np.sqrt(((xx - sx) * dx) ** 2 + ((yy - sy) * dx) ** 2)
            c_seed = 0.5 * (1.0 - np.tanh((r - sr * dx) / (np.sqrt(2) * W0)))
            c = np.maximum(c, c_seed)
            mask = r < dist_min
            dist_min = np.where(mask, r, dist_min)
            grain_theta = np.where(mask, st, grain_theta)

    cp = np.ones((Ny, Nx)); phi = np.full((Ny, Nx), phi_top)
    use_lbm = u_inlet > 0.0
    f_lbm = _lbm_init(Ny, Nx, u_inlet) if use_lbm else np.zeros((9, Ny, Nx))
    ux = np.zeros((Ny, Nx)); uy = np.zeros((Ny, Nx))

    frames = []; tip_len = []; tip_t = []
    for it in range(steps):
        if use_lbm:
            f_lbm, ux, uy = _lbm_step_jit(f_lbm, c, nu_lbm, W0, u_inlet)
        phi = _solve_phi_jit(phi, c, phi_top, phi_dep, n_phi_iter)
        new_c, new_cp, _dc = _phase_np_step_jit(
            c, cp, phi, ux, uy, grain_theta,
            dx, dt, W0, tau, delta, float(omega_aniso), beta, k_dep, alpha,
            E_theta, De, Ds, zF_RT, cs_c0, grad_phi_cap, use_lbm,
        )
        c, cp = new_c, new_cp
        if it % record_every == 0:
            col = int(cx0)
            ys = np.where(c[:, col] > 0.5)[0]
            h = ys.max() * dx if ys.size else 0.0
            tip_len.append(h); tip_t.append(it * dt)
            frames.append((it * dt, c.copy(), cp.copy(), phi.copy()))
            if verbose:
                print(f"step {it:5d}  t={it*dt:6.3f}  h={h:6.2f}  "
                      f"cmax={c.max():.2f}  cpmin={cp.min():.3f}")
    return dict(frames=frames, tip_len=np.array(tip_len),
                tip_t=np.array(tip_t), c=c, cp=cp, phi=phi)


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = run_fast(steps=4000, record_every=1000, verbose=True)
    print(f"DONE in {time.time()-t0:.1f}s. tip h={out['tip_len'][-1]:.1f}")
