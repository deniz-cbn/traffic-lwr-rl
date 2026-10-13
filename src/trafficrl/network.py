"""
Signalised grid network of LWR links.

Every link is a chain of `cells_per_link` Godunov cells. Every intersection has
4 inbound and 4 outbound links (links to/from outside the grid are entry/exit
links). Directions: E=0, W=1, S=2, N=3. Phase 0 serves E/W movements, phase 1
serves N/S movements. Intersections are handled by a conservative FIFO node
model with proportional supply sharing (docs/MATHS.md section 7).

All intersections are processed in one vectorised NumPy pass.
"""
from __future__ import annotations
import numpy as np
from .lwr import FundamentalDiagram, godunov_step, cfl_number

E, W, S, N = 0, 1, 2, 3
AXIS = np.array([0, 0, 1, 1])          # axis served by each direction (0: E/W, 1: N/S)


def turning_matrix(p_through=0.7, p_left=0.15, p_right=0.15):
    """T[d_in, d_out]: fraction of traffic arriving moving d_in that leaves moving d_out."""
    assert abs(p_through + p_left + p_right - 1) < 1e-9
    T = np.zeros((4, 4))
    right = {E: S, W: N, S: W, N: E}
    left = {E: N, W: S, S: E, N: W}
    for d in range(4):
        T[d, d] = p_through
        T[d, left[d]] = p_left
        T[d, right[d]] = p_right
    return T


class GridNetwork:
    def __init__(self, rows, cols, fd: FundamentalDiagram | None = None,
                 cells_per_link=8, dx=25.0, dt=1.0, turn=(0.7, 0.15, 0.15),
                 lost_time=2, det_cells=2):
        self.R, self.C = rows, cols
        self.n_nodes = rows * cols
        self.fd = fd or FundamentalDiagram()
        self.nc, self.dx, self.dt = cells_per_link, dx, dt
        assert cfl_number(self.fd, dt, dx) <= 1.0, "choose dt <= dx / v_f"
        self.lost_steps = int(round(lost_time / dt))
        self.det_cells = det_cells
        self.T = turning_matrix(*turn)
        self.Tpos = self.T > 0
        self._build_links()
        self.reset()

    # ------------------------------------------------------------------ topology
    def _build_links(self):
        R, C = self.R, self.C
        links = []  # (direction, src_node or -1, dst_node or -1)
        for r in range(R):
            for c in range(-1, C):                        # eastbound
                links.append((E, r * C + c if c >= 0 else -1,
                              r * C + c + 1 if c + 1 < C else -1))
            for c in range(C, -1, -1):                    # westbound
                links.append((W, r * C + c if c < C else -1,
                              r * C + c - 1 if c - 1 >= 0 else -1))
        for c in range(C):
            for r in range(-1, R):                        # southbound
                links.append((S, r * C + c if r >= 0 else -1,
                              (r + 1) * C + c if r + 1 < R else -1))
            for r in range(R, -1, -1):                    # northbound
                links.append((N, r * C + c if r < R else -1,
                              (r - 1) * C + c if r - 1 >= 0 else -1))
        self.link_dir = np.array([l[0] for l in links])
        self.link_src = np.array([l[1] for l in links])
        self.link_dst = np.array([l[2] for l in links])
        self.L = len(links)
        self.in_link = -np.ones((self.n_nodes, 4), dtype=int)    # arrives moving d
        self.out_link = -np.ones((self.n_nodes, 4), dtype=int)   # departs moving d
        for i, (d, s, t) in enumerate(links):
            if t >= 0:
                self.in_link[t, d] = i
            if s >= 0:
                self.out_link[s, d] = i
        assert (self.in_link >= 0).all() and (self.out_link >= 0).all()
        self.entry_idx = np.where(self.link_src < 0)[0]
        self.exit_idx = np.where(self.link_dst < 0)[0]

    # ------------------------------------------------------------------- state
    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rho = np.zeros((self.L, self.nc))
        self.src_queue = np.zeros(len(self.entry_idx))        # vehicles waiting outside
        self.phase = rng.integers(0, 2, self.n_nodes)
        self.t_green = rng.integers(0, 20, self.n_nodes).astype(float)
        self.clear = np.zeros(self.n_nodes, dtype=int)
        self.flux_acc = np.zeros(self.n_nodes)                # vehicles through node since last pop
        self.time = 0.0

    @property
    def mass(self):
        """Vehicles in the system (links + virtual entry queues)."""
        return float(self.rho.sum() * self.dx + self.src_queue.sum())

    # ---------------------------------------------------------------- sensors
    def detector(self):
        """(N,4) stop-line detector density / rho_max over the last det_cells cells."""
        d = self.rho[self.in_link, -self.det_cells:].mean(axis=2)
        return d / self.fd.rho_max

    def downstream_density(self):
        """(N,4) normalised density in the first cell of each outbound link."""
        return self.rho[self.out_link, 0] / self.fd.rho_max

    def green_mask(self):
        return (AXIS[None, :] == self.phase[:, None]) & (self.clear[:, None] == 0)

    def pop_flux(self):
        f = self.flux_acc.copy()
        self.flux_acc[:] = 0.0
        return f

    # ---------------------------------------------------------------- control
    def switch(self, mask):
        """Switch phase at nodes in `mask`; starts an all-red clearance period."""
        mask = np.asarray(mask, dtype=bool) & (self.clear == 0)
        self.phase = np.where(mask, 1 - self.phase, self.phase)
        self.clear = np.where(mask, self.lost_steps, self.clear)
        self.t_green = np.where(mask, 0.0, self.t_green)

    # ---------------------------------------------------------------- dynamics
    def node_flows(self, D, Sup):
        """FIFO node model. Returns (f_in[N,4], inflow_to_out[N,4])."""
        Din = D[self.in_link] * self.green_mask()
        Sout = Sup[self.out_link]
        dem_out = Din @ self.T                                # demand per outbound link
        alpha = np.minimum(1.0, Sout / np.maximum(dem_out, 1e-12))
        a_in = np.where(self.Tpos[None, :, :], alpha[:, None, :], 1.0).min(axis=2)
        f = Din * a_in                                        # actual flow per inbound
        return f, f @ self.T

    def step(self, entry_rate):
        """Advance one dt. entry_rate: (n_entry,) arrival rate [veh/s] per entry link."""
        fd, dt = self.fd, self.dt
        D = fd.demand(self.rho[:, -1])
        Sup = fd.supply(self.rho[:, 0])
        f, inn = self.node_flows(D, Sup)

        outflow = D.copy()                                    # exit links discharge freely
        outflow[self.in_link.ravel()] = f.ravel()
        inflow = np.zeros(self.L)
        inflow[self.out_link.ravel()] = inn.ravel()
        want = entry_rate + self.src_queue / dt               # arrivals + queued vehicles
        admitted = np.minimum(want, Sup[self.entry_idx])
        inflow[self.entry_idx] = admitted
        self.src_queue += (entry_rate - admitted) * dt

        self.rho, _ = godunov_step(self.rho, dt, self.dx, fd, inflow, outflow)

        # clock / phase timers
        self.flux_acc += f.sum(axis=1) * dt
        running = self.clear == 0
        self.t_green = np.where(running, self.t_green + dt, self.t_green)
        self.clear = np.maximum(self.clear - 1, 0)
        self.time += dt
        return {"exit": float(outflow[self.exit_idx].sum() * dt),
                "entered": float(admitted.sum() * dt),
                "node_flux": float(f.sum() * dt)}
