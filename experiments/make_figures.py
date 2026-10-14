"""Rebuild figures from results/results.json (no re-simulation).  python experiments/make_figures.py"""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = pathlib.Path(__file__).resolve().parents[1] / "results"
S = json.load(open(out / "results.json"))["summary"]
sizes = list(S); loads = list(S[sizes[0]]); x = np.arange(len(sizes))
fig, axs = plt.subplots(2, len(loads), figsize=(4.2 * len(loads), 7), sharex=True)
rows = [("Average junction flux: change vs sensor only (%)  [higher = better]", "fixed_vs_sensor_pct", "rl_vs_sensor_pct", "rl_vs_sensor_pct_sd"),
        ("Total time spent: change vs sensor only (%)  [lower = better]", "tts_fixed_vs_sensor_pct", "tts_rl_vs_sensor_pct", "tts_rl_vs_sensor_pct_sd")]
for r, (ylabel, kf, kr, ks) in enumerate(rows):
    for c, load in enumerate(loads):
        ax = axs[r, c]
        ax.axhline(0, color="#1f77b4", lw=1.5, label="Sensor only (baseline)")
        ax.plot(x, [S[n][load][kf] for n in sizes], "s--", color="#888888", label="Fixed-time")
        ax.errorbar(x, [S[n][load][kr] for n in sizes], yerr=[S[n][load][ks] for n in sizes], fmt="o-",
                    color="#ff7f0e", capsize=3, label="RL + sensor (mean ± sd over seeds)")
        if r == 0: ax.set_title(f"{load} load")
        if c == 0: ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{n}x{n}" for n in sizes])
        if r == 1: ax.set_xlabel("grid size")
axs[0, 0].legend(fontsize=7, loc="best")
plt.tight_layout(); plt.savefig(out / "flux_vs_sensor.png", dpi=140)
print("saved", out / "flux_vs_sensor.png")
