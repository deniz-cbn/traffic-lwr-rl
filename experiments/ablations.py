"""
Compare, on grids of increasing size, with the LWR network simulator as environment:
  fixed-time | classic actuated (sensor) | RL (pure, cold start) | RL + sensor prior (hybrid)
plus a TRANSFER test: the shared Q-table trained on a 2x2 grid is applied unchanged to every size.

    python experiments/ablations.py --episodes 50 --train-seeds 3 --eval-seeds 6
Outputs go to results/ablations/ (json, markdown table, png figures).
"""
from __future__ import annotations
import argparse, json, time, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trafficrl import GridNetwork, FixedTime, Actuated, SharedQAgent, run_episode, make_demand

T = 1800.0
LOADS = {"light": 0.25, "medium": 0.35, "heavy": 0.45}
TILT = 0.3


def evaluate(net, ctl, seeds, loads=LOADS):
    """Mean TTS per load level over eval seeds (same demand seeds for every controller)."""
    out = {}
    for name, lvl in loads.items():
        out[name] = np.array([run_episode(net, ctl, make_demand(net, lvl, TILT, s, T), T=T, seed=s)["TTS"]
                              for s in seeds])
    return out


def train(net, agent, episodes, rng, checkpoints=None, ck_fn=None):
    curve = {}
    if checkpoints and 0 in checkpoints:
        curve[0] = ck_fn(agent)
    for ep in range(episodes):
        lvl, tilt = rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.5)
        run_episode(net, agent, make_demand(net, lvl, tilt, 1000 + ep, T), T=T, train=True, seed=1000 + ep)
        agent.end_episode()
        if checkpoints and (ep + 1) in checkpoints:
            curve[ep + 1] = ck_fn(agent)
    return curve


def pct(a, b):  # % change of a relative to baseline b (negative = better)
    return 100.0 * (np.mean(a) - np.mean(b)) / np.mean(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(range(100, 100 + args.eval_seeds))
    t0 = time.time()

    # ---------------- transfer agents: trained on 2x2 only
    net2 = GridNetwork(2, 2)
    transfer = []
    for ts in range(args.train_seeds):
        ag = SharedQAgent(use_prior=True, seed=ts)
        train(net2, ag, args.episodes, np.random.default_rng(ts))
        transfer.append(ag)
    print(f"transfer agents trained ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for n in args.sizes:
        net = GridNetwork(n, n)
        res = {"fixed": evaluate(net, FixedTime(30), eval_seeds),
               "actuated": evaluate(net, Actuated(), eval_seeds)}
        for label, use_prior in (("rl_pure", False), ("rl_sensor", True)):
            per_seed = []
            for ts in range(args.train_seeds):
                ag = SharedQAgent(use_prior=use_prior, seed=ts)
                train(net, ag, args.episodes, np.random.default_rng(ts))
                per_seed.append(evaluate(net, ag, eval_seeds))
            res[label] = per_seed                       # list over training seeds
        res["rl_transfer"] = [evaluate(net, ag, eval_seeds) for ag in transfer]
        results[n] = res
        print(f"size {n}x{n} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- learning curves (cold start) on 3x3
    net3 = GridNetwork(3, 3)
    ck = [0, 2, 5, 10, 20, 40, args.episodes]
    ck = sorted(set(c for c in ck if c <= args.episodes))
    base3 = np.mean([evaluate(net3, Actuated(), eval_seeds[:4], {"m": 0.4})["m"]])
    curves = {}
    for label, use_prior in (("rl_pure", False), ("rl_sensor", True)):
        runs = []
        for ts in range(args.train_seeds):
            ag = SharedQAgent(use_prior=use_prior, seed=ts)
            fn = lambda a: float(np.mean(evaluate(net3, a, eval_seeds[:4], {"m": 0.4})["m"]) / base3)
            runs.append(train(net3, ag, args.episodes, np.random.default_rng(ts), set(ck), fn))
        curves[label] = {c: [r[c] for r in runs] for c in ck}
    print(f"learning curves done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- summarise
    summary = {}
    for n, res in results.items():
        summary[n] = {}
        for load in LOADS:
            base = res["actuated"][load]
            row = {"fixed": pct(res["fixed"][load], base), "actuated_TTS": float(np.mean(base))}
            for m in ("rl_pure", "rl_sensor", "rl_transfer"):
                per = [pct(r[load], base) for r in res[m]]       # one number per training seed
                row[m] = (float(np.mean(per)), float(np.std(per)), int(sum(p < 0 for p in per)), len(per))
            summary[n][load] = row
    json.dump({"summary": {str(k): v for k, v in summary.items()},
               "curves": {m: {str(c): v for c, v in d.items()} for m, d in curves.items()},
               "args": vars(args)}, open(out / "results.json", "w"), indent=1)

    # markdown table
    lines = ["| Grid | Load | Actuated TTS (veh·s) | Fixed-time | RL pure | RL + sensor | RL transfer (2x2→) |",
             "|---|---|---|---|---|---|---|"]
    for n, d in summary.items():
        for load, r in d.items():
            f = lambda m: f"{r[m][0]:+.1f}% ± {r[m][1]:.1f} ({r[m][2]}/{r[m][3]})"
            lines.append(f"| {n}x{n} | {load} | {r['actuated_TTS']:.0f} | {r['fixed']:+.1f}% | "
                         f"{f('rl_pure')} | {f('rl_sensor')} | {f('rl_transfer')} |")
    (out / "results.md").write_text(
        "TTS change relative to the classic actuated controller (negative = better). "
        "RL entries: mean ± std over training seeds (n of seeds that beat actuated / total).\n\n"
        + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    sizes = list(summary)
    for ax, load in zip(axs, LOADS):
        for m, lab in (("rl_pure", "RL pure"), ("rl_sensor", "RL + sensor"), ("rl_transfer", "RL transfer (2x2)")):
            mu = [summary[n][load][m][0] for n in sizes]; sd = [summary[n][load][m][1] for n in sizes]
            ax.errorbar(range(len(sizes)), mu, yerr=sd, marker="o", capsize=3, label=lab)
        ax.plot(range(len(sizes)), [summary[n][load]["fixed"] for n in sizes], "k--", marker="s", label="Fixed-time")
        ax.axhline(0, color="grey", lw=1)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{n}x{n}" for n in sizes])
        ax.set_title(f"{load} load"); ax.set_xlabel("grid size")
    axs[0].set_ylabel("TTS vs actuated (%)  — lower is better"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out / "tts_vs_size.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    for m, lab in (("rl_pure", "RL pure (cold start)"), ("rl_sensor", "RL + sensor prior")):
        xs = sorted(int(c) for c in curves[m]); arr = np.array([curves[m][c] for c in xs])
        plt.plot(xs, arr.mean(1), marker="o", label=lab); plt.fill_between(xs, arr.min(1), arr.max(1), alpha=.2)
    plt.axhline(1.0, color="k", ls="--", lw=1, label="actuated"); plt.xlabel("training episodes")
    plt.ylabel("TTS / actuated TTS (3x3, heavy)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=150)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
