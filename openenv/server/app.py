import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project root on path so all imports resolve when running locally or via Docker
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Also ensure the bess_rl package root is importable
_BESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _BESS_ROOT not in sys.path:
    sys.path.insert(0, _BESS_ROOT)

from bess_rl.openenv.models import ActionModel, ObservationModel, StepResult, ResetConfig
from bess_rl.openenv.server.env import BESSEnvironment
from bess_rl.backend.api.routes import router as api_router

app = FastAPI(
    title="BESS-RL Platform",
    version="2.0.0",
    description="OpenEnv simulation + React frontend API (SAC Agent)"
)

# Allow React dev-server (port 5173) and nginx (port 3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend API routes (/api/*)
app.include_router(api_router)

# Global environment instance (shared across requests)
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
