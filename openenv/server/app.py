import os
import sys
from fastapi import FastAPI, HTTPException
<<<<<<< HEAD

# Adjust path to import models
=======
from fastapi.middleware.cors import CORSMiddleware

# Project root on path so all bess_rl.* imports resolve
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
>>>>>>> e312b64 (initial BESS-RL commit)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bess_rl.openenv.models import ActionModel, ObservationModel, StepResult, ResetConfig
from bess_rl.openenv.server.env import BESSEnvironment
<<<<<<< HEAD

app = FastAPI(title="OpenEnv BESS Co-Optimization")
=======
from bess_rl.backend.api.routes import router as api_router

app = FastAPI(title="BESS-RL Platform", version="1.0.0",
              description="OpenEnv simulation + React frontend API")

# Allow React dev-server (port 5173) and nginx (port 3000) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend API routes  (/api/*)
app.include_router(api_router)
>>>>>>> e312b64 (initial BESS-RL commit)

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
