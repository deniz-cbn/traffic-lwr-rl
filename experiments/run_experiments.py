"""
Main study: fixed-time  vs  sensor-only (actuated)  vs  RL + sensor,
measured by the AVERAGE FLUX THROUGH THE JUNCTIONS [veh/min per junction],
on grids of increasing size and three load levels. Total time spent (TTS) is
recorded as a cross-check only.

    python experiments/run_experiments.py --episodes 50 --train-seeds 3 --eval-seeds 6
Outputs: results/results.json, results/results.md, results/flux_vs_size.png, results/learning_curve.png
(The extra RL-pure / transfer ablations live in experiments/ablations.py.)
"""
from __future__ import annotations
import argparse, json, time, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trafficrl import GridNetwork, FixedTime, Actuated, SharedQAgent, run_episode, make_demand

T = 1800.0
LOADS = {"light": 0.25, "medium": 0.35, "heavy": 0.45, "saturated": 0.65}
TILT = 0.3


def evaluate(net, ctl, seeds, loads=LOADS):
    """{load: {'flux': array over seeds, 'TTS': array over seeds}} on identical demand seeds."""
    out = {}
    for name, lvl in loads.items():
        runs = [run_episode(net, ctl, make_demand(net, lvl, TILT, s, T), T=T, seed=s) for s in seeds]
        out[name] = {"flux": np.array([r["flux"] for r in runs]), "TTS": np.array([r["TTS"] for r in runs])}
    return out


def train(net, agent, episodes, rng, checkpoints=None, ck_fn=None):
    curve = {}
    if checkpoints and 0 in checkpoints:
        curve[0] = ck_fn(agent)
    for ep in range(episodes):
        lvl, tilt = rng.uniform(0.2, 0.75), rng.uniform(0.0, 0.5)
        run_episode(net, agent, make_demand(net, lvl, tilt, 1000 + ep, T), T=T, train=True, seed=1000 + ep)
        agent.end_episode()
        if checkpoints and (ep + 1) in checkpoints:
            curve[ep + 1] = ck_fn(agent)
    return curve


def pct(a, b):
    return 100.0 * (np.mean(a) - np.mean(b)) / np.mean(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(range(100, 100 + args.eval_seeds))
    t0 = time.time()

    raw = {}
    for n in args.sizes:
        net = GridNetwork(n, n)
        res = {"fixed": evaluate(net, FixedTime(30), eval_seeds),
               "sensor": evaluate(net, Actuated(), eval_seeds), "rl": []}
        for ts in range(args.train_seeds):
            ag = SharedQAgent(use_prior=True, seed=ts)
            train(net, ag, args.episodes, np.random.default_rng(ts))
            res["rl"].append(evaluate(net, ag, eval_seeds))
        raw[n] = res
        print(f"size {n}x{n} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- learning curve on 3x3 (load 0.4): RL+sensor flux relative to sensor-only
    net3 = GridNetwork(3, 3); L = {"m": 0.4}; ev = eval_seeds[:4]
    base = np.mean(evaluate(net3, Actuated(), ev, L)["m"]["flux"])
    ck = sorted(set(c for c in [0, 1, 2, 5, 10, 20, 35, args.episodes] if c <= args.episodes))
    fn = lambda a: float(np.mean(evaluate(net3, a, ev, L)["m"]["flux"]) / base)
    runs = [train(net3, SharedQAgent(use_prior=True, seed=ts), args.episodes, np.random.default_rng(ts), set(ck), fn)
            for ts in range(args.train_seeds)]
    curve = {c: [r[c] for r in runs] for c in ck}

    # ---- summary
    summary = {}
    for n, res in raw.items():
        summary[n] = {}
        for load in LOADS:
            fx, sn = res["fixed"][load], res["sensor"][load]
            rl_flux = [r[load]["flux"] for r in res["rl"]]
            rl_tts = [r[load]["TTS"] for r in res["rl"]]
            per_f = [pct(f, sn["flux"]) for f in rl_flux]
            per_t = [pct(t, sn["TTS"]) for t in rl_tts]
            summary[n][load] = dict(
                fixed_flux=float(np.mean(fx["flux"])), sensor_flux=float(np.mean(sn["flux"])),
                sensor_flux_sd=float(np.std(sn["flux"])),
                rl_flux=float(np.mean([np.mean(f) for f in rl_flux])),
                rl_flux_seed_sd=float(np.std([np.mean(f) for f in rl_flux])),
                fixed_vs_sensor_pct=pct(fx["flux"], sn["flux"]),
                rl_vs_sensor_pct=float(np.mean(per_f)), rl_vs_sensor_pct_sd=float(np.std(per_f)),
                rl_seeds_better=int(sum(p > 0 for p in per_f)), n_train_seeds=len(per_f),
                tts_fixed_vs_sensor_pct=pct(fx["TTS"], sn["TTS"]),
                tts_rl_vs_sensor_pct=float(np.mean(per_t)), tts_rl_vs_sensor_pct_sd=float(np.std(per_t)))
    json.dump({"summary": {str(k): v for k, v in summary.items()},
               "learning_curve_flux_ratio": {str(c): v for c, v in curve.items()},
               "args": vars(args)}, open(out / "results.json", "w"), indent=1)

    L1 = ["| Grid | Load | Fixed-time | Sensor only | RL + sensor | RL vs sensor | seeds better |",
          "|---|---|---|---|---|---|---|"]
    L2 = ["| Grid | Load | Fixed-time vs sensor | RL + sensor vs sensor |", "|---|---|---|---|"]
    for n, d in summary.items():
        for load, r in d.items():
            L1.append(f"| {n}x{n} | {load} | {r['fixed_flux']:.2f} | {r['sensor_flux']:.2f} | "
                      f"{r['rl_flux']:.2f} | {r['rl_vs_sensor_pct']:+.1f}% ± {r['rl_vs_sensor_pct_sd']:.1f} | "
                      f"{r['rl_seeds_better']}/{r['n_train_seeds']} |")
            L2.append(f"| {n}x{n} | {load} | {r['tts_fixed_vs_sensor_pct']:+.1f}% | "
                      f"{r['tts_rl_vs_sensor_pct']:+.1f}% ± {r['tts_rl_vs_sensor_pct_sd']:.1f} |")
    (out / "results.md").write_text(
        "### Average flux through junctions [veh/min per junction] (higher is better)\n\n" + "\n".join(L1) +
        "\n\n### Cross-check: total time spent, change vs sensor only (negative is better)\n\n" + "\n".join(L2) + "\n")
    print("\n".join(L1)); print(); print("\n".join(L2))

    # ---- figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = list(summary); x = np.arange(len(sizes)); w = 0.27
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    for ax, load in zip(axs, LOADS):
        ax.bar(x - w, [summary[n][load]["fixed_flux"] for n in sizes], w, label="Fixed-time", color="#999999")
        ax.bar(x, [summary[n][load]["sensor_flux"] for n in sizes], w, label="Sensor only", color="#1f77b4")
        ax.bar(x + w, [summary[n][load]["rl_flux"] for n in sizes], w, label="RL + sensor", color="#ff7f0e",
               yerr=[summary[n][load]["rl_flux_seed_sd"] for n in sizes], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([f"{n}x{n}" for n in sizes]); ax.set_title(f"{load} load")
        ax.set_xlabel("grid size")
    axs[0].set_ylabel("avg flux per junction [veh/min]"); axs[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(out / "flux_vs_size.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4)); xs = sorted(curve); arr = np.array([curve[c] for c in xs])
    plt.plot(xs, arr.mean(1), marker="o", color="#ff7f0e", label="RL + sensor (mean of seeds)")
    plt.fill_between(xs, arr.min(1), arr.max(1), alpha=.2, color="#ff7f0e")
    plt.axhline(1.0, color="#1f77b4", ls="--", label="sensor only"); plt.xlabel("training episodes")
    plt.ylabel("flux / sensor-only flux (3x3, load 0.4)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=150)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
