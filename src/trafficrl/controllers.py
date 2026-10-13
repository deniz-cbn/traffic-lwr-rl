"""
Controllers. All share the interface
    reset(net); decide(net, train=False) -> bool array (n_nodes,)  [True = switch phase]
and are called every `ctrl_dt` seconds.

Sensor features are relative to the current phase: g = detector density of the
approach with most demand on the GREEN axis, r = same for the RED axis (both as
fractions of rho_max).
"""
from __future__ import annotations
import numpy as np
from .network import AXIS


def axis_demands(net):
    """(g, r): max normalised detector density on the green axis / red axis."""
    d = net.detector()
    ew, ns = d[:, :2].max(axis=1), d[:, 2:].max(axis=1)
    g = np.where(net.phase == 0, ew, ns)
    r = np.where(net.phase == 0, ns, ew)
    return g, r


class FixedTime:
    def __init__(self, green=30.0):
        self.green = green

    def reset(self, net):
        pass

    def decide(self, net, train=False):
        return (net.t_green >= self.green) & (net.clear == 0)


def actuated_rule(g, r, t_green, min_green=10.0, max_green=45.0,
                  call_thr=0.12, gap_thr=0.12):
    """Classic vehicle-actuated logic (gap-out / max-out with a call on the red side).

    Switch iff the minimum green is served, a vehicle is calling on red, and either
    the green approaches have gapped out (low density) or max green is reached.
    """
    ok = t_green >= min_green
    call = r >= call_thr
    return ok & call & ((g < gap_thr) | (t_green >= max_green))


class Actuated:
    def __init__(self, min_green=10.0, max_green=45.0, call_thr=0.12, gap_thr=0.12):
        self.kw = dict(min_green=min_green, max_green=max_green,
                       call_thr=call_thr, gap_thr=gap_thr)

    def reset(self, net):
        pass

    def decide(self, net, train=False):
        g, r = axis_demands(net)
        return actuated_rule(g, r, net.t_green, **self.kw) & (net.clear == 0)
