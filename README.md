# ⚡ BESS-RL: Battery Energy Storage System RL Environment

[![HF Space](https://img.shields.io/badge/🤗%20Space-EnergyStock-blue)](https://huggingface.co/spaces/saiteja020/EnergyStock)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://openenv.ai)

A real-world, OpenEnv-compliant reinforcement learning environment for **Battery Energy Storage System (BESS)** dispatch optimization. An agent controls a grid-scale battery to co-optimize three simultaneous revenue streams using real PJM electricity market data.

---

## What the Environment Does

The environment simulates hourly operation of a BESS connected to the PJM grid. At each timestep, the agent decides how much to charge or discharge across three objectives:

1. **Energy Arbitrage (EA):** Buy electricity when prices are low, sell when high.
2. **Frequency Regulation (FR):** Follow the PJM RegD signal to earn ancillary service revenue.
3. **Peak Shaving (PS):** Reduce net grid load below a threshold to avoid demand charge penalties.

The reward function gives **dense partial-progress signals** so agents can learn gradually — a small arbitrage win is rewarded even without full FR compliance.

---

## Tasks

| Task | Objectives | Description |
|------|-----------|-------------|
| `easy` | EA only | Learn price-arbitrage timing on PJM LMP data |
| `medium` | EA + FR | Add frequency regulation signal tracking |
| `hard` | EA + FR + PS | Full multi-objective co-optimization |

All tasks run for up to **720 hourly steps** (30 days of PJM data). Each task returns a **normalized score in [0.0, 1.0]**.

---

## Action Space

A continuous vector of **3 values**, each in `[-1.0, 1.0]`:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `a_PS` | Peak Shaving dispatch signal |
| 1 | `a_EA` | Energy Arbitrage dispatch signal |
| 2 | `a_FR` | Frequency Regulation dispatch signal |

`+1.0` = full charge, `-1.0` = full discharge. The environment combines them via `clip(a_PS + a_EA + a_FR, -1, 1)`.

---

## Observation Space

A **6-dimensional** float vector returned after each `reset()` and `step()`:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hour_of_day` | float | 0–23 | Current hour |
| `soc` | float | 0.0–1.0 | Battery State of Charge |
| `price_lmp` | float | ~0–200 | Locational Marginal Price ($/MWh) |
| `p_avg` | float | ~0–200 | 24-hour rolling average LMP ($/MWh) |
| `freq_regd` | float | -1.0–1.0 | PJM RegD frequency regulation signal |
| `load_mw` | float | ~0–50 | Grid load (MW) |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/SaiTeja020/EnergyStock
cd EnergyStock

# Install dependencies
pip install -r backend/requirements.txt
pip install openai torch numpy pandas pydantic
```

Create a `.env` file from the template:
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

---

## Running the Server

```bash
# Start the OpenEnv-compatible FastAPI server
python backend/main.py
# Server runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment, returns initial observation |
| `POST` | `/step` | Advance one timestep |
| `GET` | `/state` | Get current observation |
| `GET` | `/info` | Session metadata |

---

## Running Inference

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Run the inference script
python inference.py
```

**Expected output format:**
```
[START] task=easy env=bess-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=[0.12, -0.45, 0.33] reward=142.50 done=false error=null
[STEP] step=2 action=[0.08, -0.51, 0.29] reward=198.20 done=false error=null
...
[END] success=true steps=168 score=0.47 rewards=142.50,198.20,...
```

---

## Required Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for the LLM (OpenAI-compatible) |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face API key |

---

## Training Your Own Agent

```bash
# Train for all 3 task difficulties
python train/trainer.py --task easy --episodes 150
python train/trainer.py --task medium --episodes 300
python train/trainer.py --task hard --episodes 500
```

Model weights are saved to `train/models/`.

---

## Architecture

- **Agent:** Soft Actor-Critic (SAC) with Twin Critics and automatic entropy tuning
- **Environment:** Custom OpenEnv-compliant BESS simulation on PJM market data
- **Data:** Real PJM hourly LMP, RegD signal, and load data (auto-downloaded)
- **Export:** Models packaged as `.safetensors` for Hugging Face Hub distribution
