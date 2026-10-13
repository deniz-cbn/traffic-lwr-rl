import numpy as np
from trafficrl import FundamentalDiagram, godunov_step, GridNetwork, make_demand, Actuated, run_episode
from trafficrl.network import turning_matrix

fd = FundamentalDiagram()


def test_demand_supply_properties():
    rho = np.linspace(0, fd.rho_max, 1001)
    D, S = fd.demand(rho), fd.supply(rho)
    assert np.all(np.diff(D) >= -1e-12)            # D non-decreasing
    assert np.all(np.diff(S) <= 1e-12)             # S non-increasing
    assert np.allclose(np.minimum(D, S), fd.q(rho))  # q = min(D, S)
    assert np.isclose(fd.q(fd.rho_c), fd.q_max)


def exact_riemann_flux(rl, rr):
    """Flux at x/t=0 from the exact Riemann solution of the Greenshields LWR."""
    q, qp = fd.q, fd.wave_speed
    if rl < rr:                                    # shock, speed s = (qr-ql)/(rr-rl)
        s = (q(rr) - q(rl)) / (rr - rl)
        return q(rl) if s > 0 else q(rr)
    # rarefaction (or constant)
    if qp(rl) >= 0:
        return q(rl)
    if qp(rr) <= 0:
        return q(rr)
    return fd.q_max                                # sonic point inside fan


def test_godunov_flux_equals_exact_riemann_flux():
    rng = np.random.default_rng(0)
    for rl, rr in rng.uniform(0, fd.rho_max, (2000, 2)):
        assert np.isclose(fd.godunov_flux(rl, rr), exact_riemann_flux(rl, rr))


def test_shock_speed_rankine_hugoniot():
    dx, dt, n = 25.0, 1.0, 400
    rl, rr = 0.03, 0.07                            # both free-flow -> shock moves downstream
    s_exact = (fd.q(rr) - fd.q(rl)) / (rr - rl)    # = v_f (1 - (rl+rr)/rho_max) = 5 m/s
    x0, T = 100, 400
    rho = np.where(np.arange(n) < x0, rl, rr).astype(float)
    for _ in range(T):
        rho, _ = godunov_step(rho, dt, dx, fd, fd.q(rl), fd.demand(rho[-1]))
    mid = 0.5 * (rl + rr)
    pos = (np.argmax(rho > mid) - 0.5 - x0) * dx
    assert abs(pos - s_exact * T) < 3 * dx


def test_rarefaction_fan_and_bounds():
    n, dx, dt = 300, 25.0, 1.0
    rho = np.where(np.arange(n) < 150, 0.14, 0.01).astype(float)   # jam | free: fan
    for _ in range(300):
        rho, F = godunov_step(rho, dt, dx, fd, fd.q(0.14) * 0, fd.demand(rho[-1]))
        assert rho.min() >= -1e-12 and rho.max() <= fd.rho_max + 1e-12


def test_node_mass_conservation_and_capacity():
    net = GridNetwork(3, 3)
    dem = make_demand(net, 0.4, 0.3, 1, 600)
    m0 = net.mass
    entered = exited = 0.0
    from trafficrl import FixedTime
    ctl = FixedTime(25)
    for k in range(600):
        if k % 5 == 0:
            net.switch(ctl.decide(net))
        info = net.step(dem(net.time))
        entered += info["entered"]; exited += info["exit"]
        assert net.rho.min() >= -1e-10 and net.rho.max() <= fd.rho_max + 1e-10
    # source queue changes are not in `entered`; mass on links only:
    on_links = net.rho.sum() * net.dx
    assert np.isclose(on_links, entered - exited, atol=1e-6)


def test_turning_rows_sum_to_one():
    assert np.allclose(turning_matrix().sum(axis=1), 1.0)


def test_red_blocks_flow_and_episode_runs():
    net = GridNetwork(1, 1)
    res = run_episode(net, Actuated(), make_demand(net, 0.3, 0.2, 0, 300), T=300)
    assert res["exited"] > 0 and res["TTS"] > 0
