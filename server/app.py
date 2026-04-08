import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is on sys.path even if run from inside this directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.models import ActionModel, ObservationModel, StepResult, ResetConfig
from server.env import BESSEnvironment
from backend.api.routes import router as api_router

app = FastAPI(
    title="BESS-RL Platform",
    version="2.0.0",
    description="OpenEnv simulation + React frontend API (SAC Agent)"
)

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
_HERE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(_HERE, "..", "data", "pjm_data.csv")
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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    main()
