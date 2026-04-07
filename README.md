# BESS-RL Platform ⚡

Advanced Reinforcement Learning platform for **Battery Energy Storage System (BESS)** co-optimization. This project enables researchers and engineers to train, evaluate, and visualize RL agents performing Energy Arbitrage, Frequency Regulation, and Peak Shaving.

![BESS Visualization Mockup](https://raw.githubusercontent.com/lucide-react/lucide/main/icons/zap.svg)

## 🏗️ Architecture

The platform is built as a modular, dockerized full-stack application:

*   **Backend**: FastAPI (Python) executing high-performance RL environments and agents.
*   **Frontend**: React (Vite) dashboard with real-time Recharts telemetry.
*   **AI Engine**: Gemini 2.5 Flash for automated agent performance analysis and grading.
*   **Environment**: OpenEnv-compliant BESS simulation based on PJM market data.

---

## 🚀 Quick Start

### 1. Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
*   (Optional) Python 3.10+ for local training.

### 2. Configuration
The platform requires a Gemini API key for the AI Evaluation feature.
1. Copy the template: `cp .env.example .env`
2. Open `.env` and enter your `GEMINI_API_KEY`.

### 3. Launch
Deploy the entire stack with a single command:
```bash
docker compose up --build
```

Access the dashboard at: **[http://localhost:3000](http://localhost:3000)**

---

## ✨ Features

### 💻 Interactive Dashboard
A high-fidelity telemetry interface built for battery operational awareness.
*   **Live Charts**: Monitor State of Charge (SOC), Locational Marginal Pricing (LMP), Frequency Regulation signals, and Grid Load.
*   **Task Selection**: Toggle between task complexities:
    *   **Easy**: Energy Arbitrage only.
    *   **Medium**: EA + Frequency Regulation.
    *   **Hard**: EA + FR + Peak Shaving.
*   **Scenario Seeding**: Deterministic evaluation on specific PJM data subsets.

### 🤖 Gemini AI Evaluator
Automated expert grading of RL agents using **Gemini 2.5 Flash**.
*   **Operational Verdicts**: Qualitative assessment of agent behavior (e.g., "Good", "Needs Improvement").
*   **Strengths & Weaknesses**: Identification of specific failure modes (e.g., "Aggressive cycling", "Poor SOC management during peaks").
*   **Actionable Recommendations**: Technical suggestions for reward shaping or architecture tuning.

### 🏋️ RL Training Toolkit
The underlying engine supports the **TDD-ND (Triplet Deep Deterministic with Noise Decay)** algorithm.
*   **Delayed Policy Updates**: For stable learning.
*   **Triplet Critics**: To mitigate overestimation bias in complex reward landscapes.
*   **Customizable Rewards**: Physics-informed degradation costs and market-driven incentives.

---

## 📂 Project Structure

```text
bess_rl/
├── agent/            # RL Algorithm (Actor-Critic, Replay Buffer)
├── backend/          # FastAPI API Layer & Docker deployment
│   ├── api/          # LLM Evaluator & Frontend Routes
│   └── Dockerfile    # CPU-optimized Python image
├── data/             # PJM Market Scenarios (CSV)
├── frontend/         # React Dashboard (Vite + Recharts)
│   ├── src/          # Components, Pages, and API Client
│   └── Dockerfile    # Nginx-packaged frontend
├── openenv/          # Battery Simulator (Physics & Rewards)
├── train/            # Training & Eval scripts (Python entrypoints)
├── docker-compose.yml # Service orchestration
└── .env.example      # Environment variables template
```

---

## 🔧 Troubleshooting (Common Setup Issues)

### Docker Connection Error
If you see `error during connect: Get "...": open //./pipe/...: The system cannot find the file specified`:
*   **Ensure Docker Desktop is running**.
*   On Windows, make sure you've enabled the **WSL 2 based engine** in Docker settings.
*   Try restarting the Docker Desktop service.

### API Connection Denied
If the frontend shows "Backend Offline":
*   Check if port 8000 is occupied by another process.
*   Verify the `backend` container status: `docker compose ps`.

---

## 📜 License
This project is for educational and research purposes.

---
*Created with ❤️ for Advanced Agentic Coding.*
