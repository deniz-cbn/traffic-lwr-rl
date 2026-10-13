from __future__ import annotations
import numpy as np
from .network import AXIS


def make_demand(net, level, tilt, seed, T):
    """Time-varying entry demand [veh/s] per entry link.

    level: mean load as a fraction of link capacity q_max.
    tilt:  tidal amplitude in [0,1): E/W demand is scaled by (1 + tilt cos(2 pi t/T)),
           N/S demand by (1 - tilt cos(2 pi t/T)), plus fixed per-link noise.
    """
    rng = np.random.default_rng(10_000 + seed)
    axis = AXIS[net.link_dir[net.entry_idx]]
    noise = rng.uniform(0.85, 1.15, len(axis))
    base = level * net.fd.q_max * noise

    def demand(t):
        a = tilt * np.cos(2 * np.pi * t / T)
        return base * np.where(axis == 0, 1 + a, 1 - a)
    return demand


def run_episode(net, controller, demand, T=1800.0, ctrl_dt=5.0, train=False, seed=0):
    """Simulate T seconds. Returns metrics dict.

    TTS = total time spent [veh*s] = integral of (vehicles on links + waiting outside).
    """
    net.reset(seed)
    controller.reset(net)
    k_ctrl = int(round(ctrl_dt / net.dt))
    tts = exited = entered = 0.0
    for k in range(int(T / net.dt)):
        if k % k_ctrl == 0:
            net.switch(controller.decide(net, train))
        info = net.step(demand(net.time))
        tts += net.mass * net.dt
        exited += info["exit"]
        entered += info["entered"]
    out = dict(TTS=tts, exited=exited, remaining=net.mass, entered=entered)
    if hasattr(controller, "ep_reward"):
        out["mean_reward"] = controller.ep_reward / max(1, int(T / ctrl_dt))
    return out
