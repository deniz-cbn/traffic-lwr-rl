# Traffic flow modelling (LWR) & reinforcement-learning signal control

**Python · NumPy · FastAPI · PDEs**

* Solves the nonlinear **Lighthill–Whitham–Richards** conservation law
  $\rho_t + q(\rho)_x = 0$ with a **conservative Godunov demand–supply finite-volume scheme**.
* Builds signalised **grid networks** (1x1 junction up to city-scale grids) with a FIFO node model that
  conserves vehicles and reproduces queue spillback.
* Trains a **shared-table Q-learning** signal controller on network-wide flux rewards, using a
  compressed 96-state phase-relative state space and **sensor-informed priors** to speed up cold start.
* Research question: *does an RL light paired with the classic sensor-based (actuated) logic beat the
  sensor-based logic alone, and how does this change as junctions/networks get more complex?*

📐 **All the maths:** [`docs/MATHS.md`](docs/MATHS.md) &nbsp;|&nbsp; 📄 **Report:** [`docs/REPORT.md`](docs/REPORT.md)

## Quick start (VS Code)

```bash
git clone <your-repo-url> && cd traffic-lwr-rl
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                                  # solver & conservation tests
python experiments/run_experiments.py                   # full comparison -> results/
python experiments/train_agent.py && uvicorn api.main:app --reload   # API at /docs
```
In VS Code: select the `.venv` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*) and use the
Testing panel to run `tests/`.

## Layout
| Path | Contents |
|---|---|
| `src/trafficrl/lwr.py` | fundamental diagram, demand/supply, Godunov step |
| `src/trafficrl/network.py` | grid topology, node (intersection) model, signals, sensors |
| `src/trafficrl/controllers.py` | fixed-time and classic actuated (sensor) controllers |
| `src/trafficrl/qlearning.py` | shared-table Q-learning agent + sensor prior |
| `src/trafficrl/sim.py` | demand generator (tidal flows) and episode runner / metrics |
| `experiments/` | full study, training script |
| `tests/` | Riemann-flux equivalence, Rankine–Hugoniot shock speed, mass conservation, bounds |
| `api/main.py` | FastAPI `/simulate` endpoint |
| `docs/` | `MATHS.md`, `REPORT.md` |

## Headline results (see the report for caveats)
With the LWR network as the environment, RL + sensor reduced total time spent vs a classic actuated
controller by **7–18% at medium load and 10–19% at heavy load** (consistent over seeds at medium load), gave
**no benefit at light load**, and a table trained on a 2x2 grid transferred to grids up to 6x6 without retraining.
The sensor prior gave a safer cold start; its effect on convergence speed and stability was small and is
statistically inconclusive with 3 training seeds.
