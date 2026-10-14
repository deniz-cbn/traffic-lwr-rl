"""
Show what the shared Q-table actually learned, and where it departs from the sensor rule it was seeded with.

    python experiments/inspect_policy.py [path/to/qtable.npy]
"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficrl.qlearning import SharedQAgent, decode_state, N_STATES

G = ["empty (<0.12)", "light (0.12-0.25)", "queue (0.25-0.45)", "jam (>0.45)"]
TG = ["<20 s", "20-40 s", ">=40 s"]
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "qtable.npy"
Q = np.load(path)
ref = SharedQAgent(use_prior=True)            # untouched table = the prior (starting point)
Q0, prior = ref.Q, ref.prior
touched = np.abs(Q - Q0).max(axis=1) > 1e-9
greedy = Q.argmax(axis=1)
diff = touched & (greedy != prior)
name = ["keep", "SWITCH"]

print(f"table: {path.name} | states updated at least once: {touched.sum()}/{N_STATES} | "
      f"RL disagrees with sensor rule in {diff.sum()} of them\n")
print("| green side | red side | green time | spillback | Q(keep) | Q(switch) | sensor rule | RL choice |")
print("|---|---|---|---|---|---|---|---|")
order = np.argsort(-np.abs(Q[:, 1] - Q[:, 0]) * diff)
for s in order[: diff.sum()][:12]:
    g, r, tg, b = decode_state(s)
    print(f"| {G[g]} | {G[r]} | {TG[tg]} | {'yes' if b else 'no'} | {Q[s,0]:.2f} | {Q[s,1]:.2f} | "
          f"{name[prior[s]]} | {name[greedy[s]]} |")
