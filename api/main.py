"""
FastAPI wrapper around the simulator.

    pip install -r requirements.txt
    uvicorn api.main:app --reload        # then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import pathlib, sys
from typing import Literal
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from trafficrl import GridNetwork, FixedTime, Actuated, SharedQAgent, run_episode, make_demand

app = FastAPI(title="LWR traffic + RL signal control", version="0.1.0")
QTABLE = pathlib.Path(__file__).resolve().parents[1] / "results" / "qtable.npy"


class SimRequest(BaseModel):
    rows: int = Field(2, ge=1, le=10)
    cols: int = Field(2, ge=1, le=10)
    controller: Literal["fixed", "actuated", "rl"] = "actuated"
    load: float = Field(0.35, gt=0, le=0.6, description="mean demand as a fraction of link capacity")
    tilt: float = Field(0.3, ge=0, lt=1, description="tidal-flow amplitude")
    duration: float = Field(1800, gt=0, le=7200)
    seed: int = 0


@app.get("/health")
def health():
    return {"status": "ok", "rl_table_available": QTABLE.exists()}


@app.post("/simulate")
def simulate(req: SimRequest):
    net = GridNetwork(req.rows, req.cols)
    if req.controller == "fixed":
        ctl = FixedTime(30)
    elif req.controller == "actuated":
        ctl = Actuated()
    else:
        if not QTABLE.exists():
            raise HTTPException(404, "No trained Q-table; run `python experiments/train_agent.py` first")
        ctl = SharedQAgent(use_prior=True)
        ctl.Q = np.load(QTABLE)
    dem = make_demand(net, req.load, req.tilt, req.seed, req.duration)
    return run_episode(net, ctl, dem, T=req.duration, seed=req.seed)
