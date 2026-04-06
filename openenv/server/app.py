import os
import sys
from fastapi import FastAPI, HTTPException

# Adjust path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bess_rl.openenv.models import ActionModel, ObservationModel, StepResult, ResetConfig
from bess_rl.openenv.server.env import BESSEnvironment

app = FastAPI(title="OpenEnv BESS Co-Optimization")

# Global environment instance
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pjm_data.csv")
env = BESSEnvironment(data_path=data_path)

@app.post("/reset", response_model=ObservationModel)
def reset(config: ResetConfig):
    try:
        obs = env.reset(seed=config.seed, task=config.task)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResult)
def step(action: ActionModel):
    try:
        result = env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=ObservationModel)
def state():
    try:
        return env._get_obs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
def info():
    return {
        "task": env.task,
        "max_steps": env.max_steps,
        "current_step": env.current_step,
        "soc": env.soc
    }
