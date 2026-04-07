import os
import sys
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openenv.client import OpenEnvClient
from agent.config import AgentConfig
from agent.actor_critic import SAC_Agent

def start_server():
    import requests
    # Check if a server is already running (e.g. from Docker)
    try:
        r = requests.get("http://127.0.0.1:8000/api/health", timeout=1)
        if r.status_code == 200:
            print("Detected existing OpenEnv Server (likely Docker). Using it.")
            return None
    except:
        pass

    print("Starting local OpenEnv Server on port 8000...")
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = _ROOT
    
    server_process = subprocess.Popen(
        [sys.executable, "backend/main.py"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5) # Wait for server startup
    return server_process

def visualize(args):
    server_process = start_server()
    
    try:
        config = AgentConfig()
        client = OpenEnvClient(base_url="http://127.0.0.1:8000")
        
        agent = SAC_Agent(config)
        
        # Auto-detect trained models from the train/models/ directory
        model_path = args.model_path if args.model_path else os.path.join(_ROOT, "train", "models", f"best_model_{args.task}")
        
        if os.path.exists(model_path + "_actor.pth"):
            print(f"Loading weights from {model_path}...")
            agent.load(model_path)
        else:
            print(f"Warning: No valid SAC model found at {model_path}. Running with initialized/random weights.")
            
        print("Running an episode for evaluation...")
        state = client.reset(seed=42, task=args.task)
        done = False
        
        history = {
            "soc": [], "lmp": [], "action_ea": [],
            "action_fr": [], "r_fr": [],
            "baseline_load": [], "net_load": [],
            "action_final": []
        }
        
        max_eval_steps = args.steps if args.steps else 300
        step_count = 0
        
        while not done and step_count < max_eval_steps:
            # SAC Uses deterministic action selection for evaluation
            action = agent.select_action(np.array(state), evaluate=True)
            
            # Bound action cleanly
            action = np.clip(action, -config.max_action, config.max_action)
            next_state, reward, terminated, truncated, info = client.step(action)
            
            history["soc"].append(info["soc"])
            history["lmp"].append(info["lmp"])
            history["action_ea"].append(info["action_ea"])
            history["action_fr"].append(info["action_fr"])
            history["r_fr"].append(info["r_fr"])
            history["baseline_load"].append(info["baseline_load"])
            history["net_load"].append(info["net_load"])
            history["action_final"].append(info["action_final"])
            
            state = next_state
            done = terminated or truncated
            step_count += 1
            
        print("Generating visualization...")
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), facecolor='#111111')
        plt.subplots_adjust(hspace=0.3)
        
        # Setup dark theme aesthetic
        for ax in axes:
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, linestyle='--', color='#444444', alpha=0.5)
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.spines['right'].set_color('#888888')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            
        hours = np.arange(len(history["soc"]))
        
        # 1. State of Charge (Detailed View)
        soc_arr = np.array(history["soc"])
        axes[0].plot(hours, soc_arr, color='#2980b9', linewidth=2, label='SOC')
        # Fill areas: Green for charging, Red for discharging
        for i in range(1, len(hours)):
            color = '#27ae60' if soc_arr[i] >= soc_arr[i-1] else '#e74c3c'
            axes[0].fill_between([hours[i-1], hours[i]], [soc_arr[i-1], soc_arr[i]], color=color, alpha=0.3)
        
        axes[0].axhline(0.20, color='#f39c12', linestyle='--', linewidth=1.2, alpha=0.7, label='20% Reserve')
        axes[0].set_title("Battery State of Charge Policy", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("SOC (0.0 - 1.0)")
        axes[0].set_ylim(0, 1.1)
        axes[0].legend(loc='upper right', fontsize=9)

        # 2. Energy Arbitrage & Price Signals
        axes[1].plot(hours, history["lmp"], color='#f1c40f', label="LMP Price ($/MWh)", alpha=0.8)
        axes[1].set_ylabel("LMP ($)", color='#f1c40f')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(hours, history["action_ea"], color='#2ecc71', alpha=0.6, label="EA Dispatch Command")
        ax1_twin.set_ylabel("Dispatch [-1, 1]", color='#2ecc71')
        ax1_twin.set_ylim(-1.1, 1.1)
        ax1_twin.tick_params(colors='#2ecc71')
        axes[1].set_title("Energy Arbitrage Strategy", fontsize=14, fontweight='bold')
        
        # 3. Frequency Regulation Accuracy
        axes[2].fill_between(hours, history["r_fr"], color='#9b59b6', alpha=0.2, label="FR Earnings")
        axes[2].plot(hours, history["r_fr"], color='#9b59b6', linewidth=1, label="FR Reward")
        axes[2].set_ylabel("FR Revenue ($)", color='#9b59b6')
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(hours, history["action_fr"], color='#bdc3c7', alpha=0.5, label="RegD Command Alignment")
        ax2_twin.set_ylabel("Signal Alignment", color='#bdc3c7')
        axes[2].set_title("Frequency Regulation Performance", fontsize=14, fontweight='bold')

        # 4. Peak Shaving & Grid Impact
        axes[3].plot(hours, history["baseline_load"], color='#e74c3c', linestyle=':', label="Baseline Grid Load", alpha=0.7)
        axes[3].plot(hours, history["net_load"], color='#ecf0f1', linewidth=1.5, label="Net Grid Load (After BESS)")
        axes[3].fill_between(hours, history["baseline_load"], history["net_load"], where=(np.array(history["net_load"]) < np.array(history["baseline_load"])), color='#27ae60', alpha=0.3, label="Peak Reduction")
        axes[3].set_title("Peak Shaving Impact", fontsize=14, fontweight='bold')
        axes[3].set_ylabel("Load (MW)")
        axes[3].legend(loc="upper right", fontsize=9)
        
        output_file = "bess_visualization.png"
        fig.savefig(output_file, facecolor='#111111', bbox_inches='tight', pad_inches=0.2)
        print(f"Visualization saved to {os.path.abspath(output_file)}")
        
    finally:
        if server_process:
            print("Shutting down local server...")
            server_process.terminate()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--model-path", type=str, default=None, help="Path base name for saved agent models")
    args = parser.parse_args()
    
    visualize(args)
