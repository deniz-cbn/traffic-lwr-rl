"""
Lighthill-Whitham-Richards (LWR) model: rho_t + q(rho)_x = 0, discretised with
the conservative Godunov finite-volume scheme in demand/supply form.

See docs/MATHS.md sections 1-5 for the derivation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FundamentalDiagram:
    """Greenshields fundamental diagram q(rho) = v_f * rho * (1 - rho/rho_max).

    Units: rho [veh/m], q [veh/s], v_f [m/s].
    """
    v_f: float = 15.0        # free-flow speed (54 km/h)
    rho_max: float = 0.15    # jam density (150 veh/km/lane)

    @property
    def rho_c(self) -> float:          # critical density, argmax of q
        return 0.5 * self.rho_max

    @property
    def q_max(self) -> float:          # capacity
        return 0.25 * self.v_f * self.rho_max

    def q(self, rho):
        return self.v_f * rho * (1.0 - rho / self.rho_max)

    def wave_speed(self, rho):         # q'(rho)
        return self.v_f * (1.0 - 2.0 * rho / self.rho_max)

    def demand(self, rho):
        """Sending function D(rho): max flow a cell can SEND downstream."""
        return np.where(rho <= self.rho_c, self.q(rho), self.q_max)

    def supply(self, rho):
        """Receiving function S(rho): max flow a cell can RECEIVE from upstream."""
        return np.where(rho <= self.rho_c, self.q_max, self.q(rho))

    def godunov_flux(self, rho_l, rho_r):
        """Exact Riemann (Godunov) flux for a concave flux: min(D(rho_l), S(rho_r))."""
        return np.minimum(self.demand(rho_l), self.supply(rho_r))


def cfl_number(fd: FundamentalDiagram, dt: float, dx: float) -> float:
    return fd.v_f * dt / dx


def godunov_step(rho, dt, dx, fd: FundamentalDiagram, inflow, outflow):
    """One conservative Godunov update on the last axis of `rho`.

    rho:      (..., n) cell densities
    inflow:   (...)    flux entering the left boundary  (must be <= S(rho[...,0]))
    outflow:  (...)    flux leaving the right boundary  (must be <= D(rho[...,-1]))
    Returns (new_rho, interface_fluxes) with interface_fluxes shape (..., n+1).

        rho_i^{n+1} = rho_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    """
    assert cfl_number(fd, dt, dx) <= 1.0 + 1e-12, "CFL condition violated"
    n = rho.shape[-1]
    F = np.empty(rho.shape[:-1] + (n + 1,), dtype=float)
    F[..., 0] = inflow
    F[..., n] = outflow
    if n > 1:
        F[..., 1:n] = fd.godunov_flux(rho[..., :-1], rho[..., 1:])
    return rho + (dt / dx) * (F[..., :-1] - F[..., 1:]), F
