"""
Live / GIF visualisation of the LWR network under different controllers, side by side on IDENTICAL demand.

    python experiments/animate.py                       # saves results/simulation.gif
    python experiments/animate.py --show                # opens an interactive window instead
    python experiments/animate.py --size 4 --load 0.5 --duration 900 --controllers actuated rl

Link colour = LWR density / jam density (green = free flow, red = jammed).
Dots at stop lines = signal state (green / red). Each cell of every link is drawn separately,
so you can see queues build up and shock waves travel backwards along the links.
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficrl import GridNetwork, FixedTime, Actuated, SharedQAgent, run_episode, make_demand

DIRV = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], float)       # E, W, S, N on screen (y up)
LAT = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], float)        # right-hand-traffic lateral offset


def geometry(net, gap=0.14, lat=0.055):
    node_xy = np.array([[c, -r] for r in range(net.R) for c in range(net.C)], float)
    P0, P1 = np.zeros((net.L, 2)), np.zeros((net.L, 2))
    for l in range(net.L):
        d = net.link_dir[l]; v = DIRV[d]
        s, t = net.link_src[l], net.link_dst[l]
        a = node_xy[s] if s >= 0 else node_xy[t] - v
        b = node_xy[t] if t >= 0 else node_xy[s] + v
        if s >= 0: a = a + v * gap
        if t >= 0: b = b - v * gap
        P0[l], P1[l] = a + lat * LAT[d], b + lat * LAT[d]
    f = np.linspace(0, 1, net.nc + 1)
    segs = np.empty((net.L, net.nc, 2, 2))
    for i in range(net.nc):
        segs[:, i, 0] = P0 + (P1 - P0) * f[i]
        segs[:, i, 1] = P0 + (P1 - P0) * f[i + 1]
    stop_xy = P1[net.in_link]                                       # (N,4,2) stop-line positions
    return node_xy, segs.reshape(-1, 2, 2), stop_xy


def get_rl_agent(episodes=40):
    """Load results/qtable.npy, or train a quick shared table on a 2x2 grid (it transfers to any size)."""
    ag = SharedQAgent(use_prior=True, seed=0)
    qpath = ROOT / "results" / "qtable.npy"
    if qpath.exists():
        ag.Q = np.load(qpath); return ag
    print(f"No trained Q-table found: training {episodes} quick episodes on a 2x2 grid ...")
    net, T, rng = GridNetwork(2, 2), 1800.0, np.random.default_rng(0)
    for ep in range(episodes):
        run_episode(net, ag, make_demand(net, rng.uniform(.2, .75), rng.uniform(0, .5), 1000 + ep, T),
                    T=T, train=True, seed=1000 + ep)
        ag.end_episode()
    qpath.parent.mkdir(exist_ok=True); np.save(qpath, ag.Q)
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--load", type=float, default=0.45)
    ap.add_argument("--tilt", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=600.0, help="simulated seconds")
    ap.add_argument("--controllers", nargs="+", default=["fixed", "actuated", "rl"],
                    choices=["fixed", "actuated", "rl"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--save", default=str(ROOT / "results" / "simulation.gif"))
    ap.add_argument("--show", action="store_true", help="open an interactive window instead of saving")
    a = ap.parse_args()

    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FuncAnimation
    from matplotlib import cm, colors

    labels = {"fixed": "Fixed-time", "actuated": "Sensor only", "rl": "RL + sensor"}
    make = {"fixed": lambda: FixedTime(30), "actuated": Actuated, "rl": lambda: get_rl_agent()}
    ctls = {k: make[k]() for k in a.controllers}
    nets = {k: GridNetwork(a.size, a.size) for k in a.controllers}
    demand = make_demand(nets[a.controllers[0]], a.load, a.tilt, a.seed, a.duration)
    ctrl_dt, k_ctrl = 5.0, 5
    n_frames = int(a.duration / ctrl_dt)
    node_xy, segs, stop_xy = geometry(nets[a.controllers[0]])
    cmap = plt.get_cmap("RdYlGn_r")

    n = len(a.controllers)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n + 0.6, 5.0), squeeze=False)
    axs = axs[0]
    art = {}
    pad = 0.8
    for ax, k in zip(axs, a.controllers):
        lc = LineCollection(segs, linewidths=3.2, cmap=cmap, norm=colors.Normalize(0, 1), capstyle="butt")
        lc.set_array(np.zeros(len(segs))); ax.add_collection(lc)
        ax.scatter(node_xy[:, 0], node_xy[:, 1], marker="s", s=120, c="#444444", zorder=2)
        dots = ax.scatter(stop_xy[:, :, 0].ravel(), stop_xy[:, :, 1].ravel(), s=26, c="red",
                          edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlim(-pad, a.size - 1 + pad); ax.set_ylim(-(a.size - 1) - pad, pad)
        ax.set_aspect("equal"); ax.axis("off")
        title = ax.set_title(labels[k], fontsize=12, fontweight="bold")
        txt = ax.text(0.5, -0.04, "", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        art[k] = dict(lc=lc, dots=dots, txt=txt, flux=0.0)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap)
    cax = fig.add_axes([0.935, 0.28, 0.012, 0.44])
    fig.colorbar(sm, cax=cax, label="density / jam density")
    fig.suptitle(f"LWR network {a.size}x{a.size}  |  load {a.load} of capacity  |  tidal demand", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.92, wspace=0.03, top=0.90, bottom=0.10)

    def reset_all():
        for k in a.controllers:
            nets[k].reset(a.seed); ctls[k].reset(nets[k]); art[k]["flux"] = 0.0

    def update(frame):
        if frame == 0:
            reset_all()
        out = []
        for k in a.controllers:
            net, ctl = nets[k], ctls[k]
            net.switch(ctl.decide(net, False))
            for _ in range(k_ctrl):
                info = net.step(demand(net.time))
                art[k]["flux"] += info["node_flux"]
            art[k]["lc"].set_array((net.rho / net.fd.rho_max).ravel())
            green = net.green_mask()
            art[k]["dots"].set_facecolors(np.where(green.ravel()[:, None], [[0.1, 0.8, 0.1, 1]], [[0.9, 0.1, 0.1, 1]]))
            avg = art[k]["flux"] / (net.time / 60.0) / net.n_nodes
            art[k]["txt"].set_text(f"t = {net.time:4.0f} s   vehicles = {net.mass:5.0f}\n"
                                   f"avg flux = {avg:5.1f} veh/min/junction")
            out += [art[k]["lc"], art[k]["dots"], art[k]["txt"]]
        return out

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / a.fps, blit=False, repeat=True)
    if a.show:
        plt.show()
    else:
        pathlib.Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        ani.save(a.save, writer="pillow", fps=a.fps, dpi=75)
        print("saved", a.save)


if __name__ == "__main__":
    main()
