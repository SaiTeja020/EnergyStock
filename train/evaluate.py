import os
import sys
import time
import subprocess
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bess_rl.openenv.client import OpenEnvClient
from bess_rl.agent.config import AgentConfig
from bess_rl.agent.actor_critic import TDD_ND_Agent


def start_server():
    import requests
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_file = open("server_eval.log", "w")
    server_process = subprocess.Popen(
        ["uvicorn", "bess_rl.openenv.server.app:app", "--port", "8000", "--log-level", "warning"],
        env=env_vars, stdout=log_file, stderr=log_file
    )
    # Poll until server is ready (up to 20 seconds)
    for attempt in range(20):
        time.sleep(1)
        try:
            r = requests.get("http://127.0.0.1:8000/docs", timeout=1)
            if r.status_code < 500:
                # print(f"  Server ready after {attempt+1}s")
                break
        except Exception:
            pass
    else:
        print("  WARNING: Server may not be ready. Check server_eval.log for errors.")
    return server_process


def evaluate_model(client, model_path, task, seeds):
    config = AgentConfig()
    agent = TDD_ND_Agent(config)

    # if os.path.exists(model_path + "_actor.pth"):
    #     agent.load(model_path)
    #     print(f"  Loaded: {model_path}")
    # else:
    #     print(f"  WARNING: No model at {model_path}. Using random weights.")
    if os.path.exists(model_path + "_actor.pth"):
        agent.load(model_path)

    rewards, soc_at_peak_hrs, peak_violation_rates = [], [], []
    cycle_counts, arb_accuracies, fr_scores = [], [], []

    for seed in seeds:
        state = client.reset(seed=seed, task=task)
        done = False
        ep_reward = 0
        soc_hist, hour_hist = [], []
        violations = total_steps = action_dir_correct = 0
        fr_score_sum = fr_eligible = direction_changes = 0
        prev_soc = None

        while not done:
            action = np.clip(agent.select_action(np.array(state)), -config.max_action, config.max_action)
            next_state, reward, terminated, truncated, info = client.step(action)

            ep_reward += reward
            total_steps += 1

            soc      = info["soc"]
            net_load = info["net_load"]
            lmp      = info["lmp"]
            r_fr     = info["r_fr"]
            p_avg    = float(next_state[3]) if len(next_state) > 3 else lmp
            hour     = int(float(state[0]))

            soc_hist.append(soc)
            hour_hist.append(hour)

            if net_load > 20.0:
                violations += 1

            # Arbitrage direction accuracy
            price_signal  = lmp - p_avg
            action_final  = info["action_final"]
            if price_signal > 1.0 and action_final < 0:    # High price → should discharge
                action_dir_correct += 1
            elif price_signal < -1.0 and action_final > 0: # Low price → should charge
                action_dir_correct += 1
            elif abs(price_signal) <= 1.0:                  # Neutral zone → any action ok
                action_dir_correct += 1

            if r_fr > 0:
                fr_score_sum += r_fr
                fr_eligible  += 1

            if prev_soc is not None and prev_soc != soc:
                if (soc > prev_soc) != (prev_soc > 0.5):
                    direction_changes += 1
            prev_soc = soc

            state = next_state
            done  = terminated or truncated

        rewards.append(ep_reward)
        peak_violation_rates.append(violations / total_steps * 100)

        peak_soc = [soc_hist[i] for i, h in enumerate(hour_hist) if 16 <= h <= 20]
        if peak_soc:
            soc_at_peak_hrs.append(np.mean(peak_soc))

        cycle_counts.append(direction_changes / 2.0)
        arb_accuracies.append(action_dir_correct / total_steps * 100)
        fr_scores.append(fr_score_sum / max(fr_eligible, 1))

    return {
        "reward_mean":        np.mean(rewards),
        "reward_std":         np.std(rewards),
        "reward_min":         np.min(rewards),
        "reward_max":         np.max(rewards),
        "soc_at_peak_mean":   np.mean(soc_at_peak_hrs) if soc_at_peak_hrs else 0.0,
        "peak_violation_pct": np.mean(peak_violation_rates),
        "avg_cycles_per_ep":  np.mean(cycle_counts),
        "arb_accuracy_pct":   np.mean(arb_accuracies),
        "avg_fr_score_per_hit": np.mean(fr_scores),
    }


def score_model(results, task):
    s = {}
    ceilings = {"easy": 160000, "medium": 185000, "hard": 190000}

    s["reward"]           = min(1.0, results["reward_mean"] / ceilings[task])
    s["soc_readiness"]    = min(1.0, results["soc_at_peak_mean"] / 0.75)
    s["ps_adherence"]     = max(0.0, 1.0 - results["peak_violation_pct"] / 20.0)
    s["cycle_discipline"] = max(0.0, 1.0 - results["avg_cycles_per_ep"] / 200.0)
    s["arb_accuracy"]     = max(0.0, (results["arb_accuracy_pct"] - 50.0) / 50.0)
    cv = results["reward_std"] / max(abs(results["reward_mean"]), 1)
    s["consistency"]      = max(0.0, 1.0 - cv * 3)

    if task == "easy":
        w = {"reward": 0.35, "soc_readiness": 0.25, "ps_adherence": 0.00,
             "cycle_discipline": 0.15, "arb_accuracy": 0.20, "consistency": 0.05}
    elif task == "medium":
        w = {"reward": 0.30, "soc_readiness": 0.20, "ps_adherence": 0.00,
             "cycle_discipline": 0.15, "arb_accuracy": 0.20, "consistency": 0.15}
    else:
        w = {"reward": 0.25, "soc_readiness": 0.15, "ps_adherence": 0.20,
             "cycle_discipline": 0.15, "arb_accuracy": 0.15, "consistency": 0.10}

    total = sum(s[k] * w[k] for k in w)
    return s, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=str, default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--seeds", type=int, default=20, help="Number of evaluation seeds (starting from 300)")
    args = parser.parse_args()

    eval_seeds = list(range(300, 300 + args.seeds))
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print(f"\n{'='*60}")
    print(f"  BESS-RL Evaluation  |  Seeds {eval_seeds[0]}-{eval_seeds[-1]}  (unseen)")
    print(f"{'='*60}\n")
    print("  NOTE: Ensure the OpenEnv server is already running:")
    print("  > uvicorn bess_rl.openenv.server.app:app --port 8000")
    print()

    client = OpenEnvClient(base_url="http://127.0.0.1:8000")

    for task in tasks:
        model_path = f"models/best_model_{task}"
        print(f"Evaluating [{task.upper()}] model on {len(eval_seeds)} seeds...")
        results = evaluate_model(client, model_path, task, eval_seeds)
        scores, overall = score_model(results, task)

        print(f"\n  --- {task.upper()} Results ---")
        print(f"  Reward:        mean={results['reward_mean']:>10.0f}  std={results['reward_std']:>8.0f}"
              f"  min={results['reward_min']:>10.0f}  max={results['reward_max']:>10.0f}")
        print(f"  SOC at Peak:   {results['soc_at_peak_mean']:.1%}  (target >70%)")
        print(f"  PS Violations: {results['peak_violation_pct']:.1f}%  (target <5%)")
        print(f"  Avg Cycles:    {results['avg_cycles_per_ep']:.0f}  per episode")
        print(f"  Arb Accuracy:  {results['arb_accuracy_pct']:.1f}%  (50%=random, 100%=perfect)")
        print(f"\n  --- Dimension Scores ---")
        for dim, sc in scores.items():
            bar = '█' * int(sc * 20) + '░' * (20 - int(sc * 20))
            print(f"  {dim:<20} [{bar}] {sc:.2f}")
        print(f"\n  ★ OVERALL SCORE: {overall:.3f} / 1.000\n")
        print(f"{'='*60}\n")
