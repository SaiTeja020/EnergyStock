"""
API routes consumed by the React frontend.
"""
import os
import sys
import glob
import numpy as np
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bess_rl.agent.config import AgentConfig
from bess_rl.agent.actor_critic import SAC_Agent
from bess_rl.openenv.server.env import BESSEnvironment
from bess_rl.openenv.models import ActionModel

router = APIRouter(prefix="/api", tags=["frontend"])

_DATA_PATH = os.path.join(_ROOT, "data", "pjm_data.csv")
_MODELS_DIR = os.path.join(_ROOT, "train", "models")


def _load_agent(model_base: str) -> SAC_Agent:
    config = AgentConfig()
    agent = SAC_Agent(config)
    if os.path.exists(model_base + "_actor.pth"):
        agent.load(model_base)
    return agent


# ── Schemas ──────────────────────────────────────────────────────────────────

class RunEpisodeRequest(BaseModel):
    task: str = "hard"
    seed: int = 42
    model_name: str = "best_model_hard"
    max_steps: Optional[int] = 300


class EvaluateRequest(BaseModel):
    task: str = "hard"
    model_name: str = "best_model_hard"
    num_seeds: int = 10
    seed_start: int = 300


class EpisodeStep(BaseModel):
    step: int
    soc: float
    lmp: float
    action_ea: float
    action_fr: float
    action_ps: float
    action_final: float
    r_ea: float
    r_fr: float
    r_ps: float
    reward: float
    baseline_load: float
    net_load: float


class RunEpisodeResponse(BaseModel):
    task: str
    model_name: str
    seed: int
    total_reward: float
    steps: List[EpisodeStep]


class ScoreBreakdown(BaseModel):
    reward: float
    soc_readiness: float
    ps_adherence: float
    cycle_discipline: float
    arb_accuracy: float
    consistency: float
    overall: float


class EvaluateResponse(BaseModel):
    task: str
    model_name: str
    num_seeds: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    soc_at_peak_mean: float
    peak_violation_pct: float
    avg_cycles_per_ep: float
    arb_accuracy_pct: float
    avg_fr_score_per_hit: float
    scores: ScoreBreakdown


class LLMAnalysisRequest(BaseModel):
    evaluation: EvaluateResponse


class LLMAnalysisResponse(BaseModel):
    available: bool
    verdict: Optional[str] = None
    summary: Optional[str] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    confidence: Optional[str] = None
    detailed_analysis: Optional[str] = None
    error: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "service": "BESS-RL Backend"}


@router.get("/models")
def list_models():
    if not os.path.isdir(_MODELS_DIR):
        return {"models": []}
    actor_files = glob.glob(os.path.join(_MODELS_DIR, "*_actor.pth"))
    names = sorted({os.path.basename(p).replace("_actor.pth", "") for p in actor_files})
    return {"models": names}


@router.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "label": "Easy",   "description": "Energy Arbitrage only"},
            {"id": "medium", "label": "Medium",  "description": "Energy Arbitrage + Frequency Regulation"},
            {"id": "hard",   "label": "Hard",    "description": "Energy Arbitrage + FR + Peak Shaving"},
        ]
    }


@router.post("/run-episode", response_model=RunEpisodeResponse)
def run_episode(req: RunEpisodeRequest):
    model_base = os.path.join(_MODELS_DIR, req.model_name)
    try:
        agent = _load_agent(model_base)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent: {e}")

    config = AgentConfig()
    env = BESSEnvironment(data_path=_DATA_PATH)

    obs = env.reset(seed=req.seed, task=req.task)
    state = np.array([obs.hour_of_day, obs.soc, obs.price_lmp,
                      obs.p_avg, obs.freq_regd, obs.load_mw], dtype=np.float32)

    steps: List[EpisodeStep] = []
    total_reward = 0.0
    done = False
    step_idx = 0
    max_steps = req.max_steps or env.max_steps

    while not done and step_idx < max_steps:
        action = np.clip(agent.select_action(state), -config.max_action, config.max_action)
        result = env.step(ActionModel(action=action.tolist()))

        info = result.info
        total_reward += result.reward
        steps.append(EpisodeStep(
            step=step_idx,
            soc=info["soc"], lmp=info["lmp"],
            action_ea=info["action_ea"], action_fr=info["action_fr"],
            action_ps=info["action_ps"], action_final=info["action_final"],
            r_ea=info["r_ea"], r_fr=info["r_fr"], r_ps=info["r_ps"],
            reward=result.reward,
            baseline_load=info["baseline_load"], net_load=info["net_load"],
        ))

        o = result.observation
        state = np.array([o.hour_of_day, o.soc, o.price_lmp,
                          o.p_avg, o.freq_regd, o.load_mw], dtype=np.float32)
        done = result.terminated or result.truncated
        step_idx += 1

    return RunEpisodeResponse(task=req.task, model_name=req.model_name,
                              seed=req.seed, total_reward=total_reward, steps=steps)


def _compute_scores(results: dict, task: str) -> dict:
    ceilings = {"easy": 160000, "medium": 185000, "hard": 190000}
    s = {
        "reward":           min(1.0, results["reward_mean"] / ceilings[task]),
        "soc_readiness":    min(1.0, results["soc_at_peak_mean"] / 0.75),
        "ps_adherence":     max(0.0, 1.0 - results["peak_violation_pct"] / 20.0),
        "cycle_discipline": max(0.0, 1.0 - results["avg_cycles_per_ep"] / 200.0),
        "arb_accuracy":     max(0.0, (results["arb_accuracy_pct"] - 50.0) / 50.0),
        "consistency":      max(0.0, 1.0 - (results["reward_std"] / max(abs(results["reward_mean"]), 1)) * 3),
    }
    weights = {
        "easy":   {"reward": 0.35, "soc_readiness": 0.25, "ps_adherence": 0.00, "cycle_discipline": 0.15, "arb_accuracy": 0.20, "consistency": 0.05},
        "medium": {"reward": 0.30, "soc_readiness": 0.20, "ps_adherence": 0.00, "cycle_discipline": 0.15, "arb_accuracy": 0.20, "consistency": 0.15},
        "hard":   {"reward": 0.25, "soc_readiness": 0.15, "ps_adherence": 0.20, "cycle_discipline": 0.15, "arb_accuracy": 0.15, "consistency": 0.10},
    }
    w = weights[task]
    s["overall"] = sum(s[k] * w[k] for k in w)
    return s


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    model_base = os.path.join(_MODELS_DIR, req.model_name)
    config = AgentConfig()
    try:
        agent = _load_agent(model_base)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent: {e}")

    env = BESSEnvironment(data_path=_DATA_PATH)
    seeds = list(range(req.seed_start, req.seed_start + req.num_seeds))

    rewards, soc_peak, violations_pct, cycles, arb_acc, fr_sc = [], [], [], [], [], []

    for seed in seeds:
        obs = env.reset(seed=seed, task=req.task)
        state = np.array([obs.hour_of_day, obs.soc, obs.price_lmp,
                          obs.p_avg, obs.freq_regd, obs.load_mw], dtype=np.float32)
        done = False
        ep_reward = 0.0
        soc_hist, hour_hist = [], []
        viol = total = dir_ok = fr_sum = fr_elig = dir_changes = 0
        prev_soc = None

        while not done:
            action = np.clip(agent.select_action(state), -config.max_action, config.max_action)
            result = env.step(ActionModel(action=action.tolist()))
            info = result.info
            ep_reward += result.reward
            total += 1

            soc = info["soc"]; hour = int(float(state[0]))
            soc_hist.append(soc); hour_hist.append(hour)

            if info["net_load"] > 20.0: viol += 1
            ps = info["lmp"] - float(state[3])
            af = info["action_final"]
            if (ps > 1.0 and af < 0) or (ps < -1.0 and af > 0) or abs(ps) <= 1.0: dir_ok += 1
            if info["r_fr"] > 0: fr_sum += info["r_fr"]; fr_elig += 1
            if prev_soc is not None and prev_soc != soc:
                if (soc > prev_soc) != (prev_soc > 0.5): dir_changes += 1
            prev_soc = soc

            o = result.observation
            state = np.array([o.hour_of_day, o.soc, o.price_lmp,
                              o.p_avg, o.freq_regd, o.load_mw], dtype=np.float32)
            done = result.terminated or result.truncated

        rewards.append(ep_reward)
        violations_pct.append(viol / total * 100)
        pk = [soc_hist[i] for i, h in enumerate(hour_hist) if 16 <= h <= 20]
        if pk: soc_peak.append(float(np.mean(pk)))
        cycles.append(dir_changes / 2.0)
        arb_acc.append(dir_ok / total * 100)
        fr_sc.append(fr_sum / max(fr_elig, 1))

    res = {
        "reward_mean":          float(np.mean(rewards)),
        "reward_std":           float(np.std(rewards)),
        "reward_min":           float(np.min(rewards)),
        "reward_max":           float(np.max(rewards)),
        "soc_at_peak_mean":     float(np.mean(soc_peak)) if soc_peak else 0.0,
        "peak_violation_pct":   float(np.mean(violations_pct)),
        "avg_cycles_per_ep":    float(np.mean(cycles)),
        "arb_accuracy_pct":     float(np.mean(arb_acc)),
        "avg_fr_score_per_hit": float(np.mean(fr_sc)),
    }
    scores = _compute_scores(res, req.task)
    return EvaluateResponse(task=req.task, model_name=req.model_name,
                            num_seeds=req.num_seeds, **res,
                            scores=ScoreBreakdown(**scores))


@router.post("/llm-analyze", response_model=LLMAnalysisResponse)
def llm_analyze(req: LLMAnalysisRequest):
    from bess_rl.backend.api.llm_evaluator import get_llm_analysis
    result = get_llm_analysis(req.evaluation.model_dump())
    return LLMAnalysisResponse(**result)
