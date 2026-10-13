"""
Shared-table tabular Q-learning for signal control.

* ONE Q-table shared by every intersection (parameter sharing): each decision at each
  intersection is a sample for the same table, so data grows with network size and the
  learned policy transfers to larger networks.
* Compressed, phase-relative local state (96 states): see `encode`.
* Reward: blend of network-wide and local normalised flux through intersections.
* Sensor-informed prior (optional): Q initialised to favour the classic actuated rule,
  and exploration is guided towards that rule (docs/MATHS.md section 10).
"""
from __future__ import annotations
import numpy as np
from .controllers import axis_demands, actuated_rule

DENS_EDGES = np.array([0.12, 0.25, 0.45])       # bins of rho/rho_max -> 0..3
TG_EDGES = np.array([20.0, 40.0])               # bins of green time [s] -> 0..2
DENS_REP = np.array([0.05, 0.18, 0.35, 0.60])   # representative values (for the prior)
TG_REP = np.array([15.0, 30.0, 45.0])
BLOCK_THR = 0.6
N_STATES, N_ACTIONS = 4 * 4 * 3 * 2, 2          # 96 x {keep, switch}


def state_index(g_bin, r_bin, tg_bin, blocked):
    return ((g_bin * 4 + r_bin) * 3 + tg_bin) * 2 + blocked


def decode_state(s):
    blocked = s % 2; s //= 2
    tg = s % 3; s //= 3
    r = s % 4; g = s // 4
    return g, r, tg, blocked


def sensor_prior_actions(**kw):
    """Actuated rule evaluated at the representative value of every discretised state."""
    pri = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        g, r, tg, _ = decode_state(s)
        pri[s] = int(actuated_rule(DENS_REP[g], DENS_REP[r], TG_REP[tg], **kw))
    return pri


class SharedQAgent:
    def __init__(self, use_prior=True, alpha=0.1, gamma=0.9, eps0=0.3, eps_min=0.02,
                 eps_decay=0.97, guide_p=0.7, q_init=6.0, prior_bonus=0.3, lam=0.5,
                 min_green=10.0, max_green=45.0, call_thr=0.12, ctrl_dt=5.0, seed=0):
        self.use_prior, self.alpha, self.gamma = use_prior, alpha, gamma
        self.eps0, self.eps_min, self.eps_decay = eps0, eps_min, eps_decay
        self.guide_p, self.lam = guide_p, lam
        self.min_green, self.max_green, self.call_thr = min_green, max_green, call_thr
        self.ctrl_dt = ctrl_dt
        self.rng = np.random.default_rng(seed)
        self.prior = sensor_prior_actions(min_green=min_green, max_green=max_green,
                                          call_thr=call_thr, gap_thr=call_thr)
        self.Q = np.full((N_STATES, N_ACTIONS), float(q_init))
        if use_prior:
            self.Q[np.arange(N_STATES), self.prior] += prior_bonus
        self.episode = 0
        self.eps = eps0
        self.reset_episode_stats()

    # -------------------------------------------------------------- bookkeeping
    def reset_episode_stats(self):
        self.prev = None
        self.ep_reward, self.ep_updates = 0.0, 0

    def reset(self, net):
        self.reset_episode_stats()
        self.q_norm = 2.0 * net.fd.q_max * self.ctrl_dt     # max vehicles through a node/interval

    def end_episode(self):
        self.episode += 1
        self.eps = max(self.eps_min, self.eps0 * self.eps_decay ** self.episode)

    # ----------------------------------------------------------------- features
    def encode(self, net):
        g, r = axis_demands(net)
        g_bin = np.searchsorted(DENS_EDGES, g, side="right")
        r_bin = np.searchsorted(DENS_EDGES, r, side="right")
        tg_bin = np.searchsorted(TG_EDGES, net.t_green, side="right")
        blocked = (net.downstream_density().max(axis=1) > BLOCK_THR).astype(int)
        return state_index(g_bin, r_bin, tg_bin, blocked), g, r

    # ------------------------------------------------------------------ control
    def decide(self, net, train=False):
        s2, g, r = self.encode(net)
        n = len(s2)

        # --- learn from the transition that just finished
        if self.prev is not None:
            s, a, free = self.prev
            flux = net.pop_flux() / self.q_norm
            reward = self.lam * flux.mean() + (1 - self.lam) * flux
            self.ep_reward += reward.mean()
            if train:
                for i in np.where(free)[0]:
                    target = reward[i] + self.gamma * self.Q[s2[i]].max()
                    self.Q[s[i], a[i]] += self.alpha * (target - self.Q[s[i], a[i]])
                    self.ep_updates += 1
        else:
            net.pop_flux()

        # --- safety layer (min/max green) is the same sensor logic as the baseline
        in_clear = net.clear > 0
        forced_keep = (net.t_green < self.min_green) | in_clear
        forced_switch = (~forced_keep) & (net.t_green >= self.max_green) & (r >= self.call_thr)
        free = ~(forced_keep | forced_switch)

        greedy = self.Q[s2].argmax(axis=1)
        a = greedy.copy()
        if train:
            explore = self.rng.random(n) < self.eps
            rand_a = self.rng.integers(0, 2, n)
            if self.use_prior:
                guided = self.rng.random(n) < self.guide_p
                rand_a = np.where(guided, self.prior[s2], rand_a)
            a = np.where(explore, rand_a, greedy)
        a = np.where(forced_keep, 0, np.where(forced_switch, 1, a))
        self.prev = (s2, a, free)
        return a.astype(bool)
