"""Train one shared Q-table (sensor prior) and save it to results/qtable.npy (used by the API)."""
import sys, pathlib, argparse
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trafficrl import GridNetwork, SharedQAgent, run_episode, make_demand

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=2)
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
net, T = GridNetwork(a.size, a.size), 1800.0
agent, rng = SharedQAgent(use_prior=True, seed=a.seed), np.random.default_rng(a.seed)
for ep in range(a.episodes):
    lvl, tilt = rng.uniform(0.2, 0.5), rng.uniform(0, 0.5)
    r = run_episode(net, agent, make_demand(net, lvl, tilt, 1000 + ep, T), T=T, train=True, seed=1000 + ep)
    agent.end_episode()
    if (ep + 1) % 10 == 0:
        print(f"episode {ep+1:4d}  eps={agent.eps:.3f}  TTS={r['TTS']:.0f}  mean reward={r['mean_reward']:.3f}")
out = pathlib.Path(__file__).resolve().parents[1] / "results" / "qtable.npy"
np.save(out, agent.Q); print("saved", out)
