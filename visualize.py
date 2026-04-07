import os
import sys
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bess_rl.openenv.client import OpenEnvClient
from bess_rl.agent.config import AgentConfig
from bess_rl.agent.actor_critic import TDD_ND_Agent

def start_server():
    print("Starting OpenEnv Server on port 8000...")
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    server_process = subprocess.Popen(
        ["uvicorn", "bess_rl.openenv.server.app:app", "--port", "8000"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3) # Wait for server startup
    return server_process

def visualize(args):
    server_process = start_server()
    
    try:
        config = AgentConfig()
        client = OpenEnvClient(base_url="http://127.0.0.1:8000")
        
        agent = TDD_ND_Agent(config)
        
        # Optionally load a pretrained model if provided
        if args.model_path and os.path.exists(args.model_path + "_actor.pth"):
            print(f"Loading weights from {args.model_path}...")
            agent.load(args.model_path)
        else:
            print("No model path provided or found. Running with initialized/random weights.")
            
        print("Running an episode for evaluation...")
        state = client.reset(seed=42, task=args.task)
        done = False
        
        history = {
            "soc": [], "lmp": [], "action_ea": [],
            "action_fr": [], "r_fr": [],
            "baseline_load": [], "net_load": []
        }
        
        # We simulate a specific length (e.g. 300 hours) to match the reference image style
        max_eval_steps = args.steps if args.steps else 300
        step_count = 0
        
        while not done and step_count < max_eval_steps:
            # Deterministic evaluation (no noise)
            action = agent.select_action(np.array(state))
            
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
            
            state = next_state
            done = terminated or truncated
            step_count += 1
            
        print("Generating visualization...")
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), facecolor='#111111')
        plt.subplots_adjust(hspace=0.3)
        
        # Setup common dark theme plot attributes to match the image exactly
        for ax in axes:
            ax.set_facecolor('white')
            ax.grid(True, linestyle='-', alpha=0.3)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            
        x_range = np.arange(len(history["soc"]))
        
<<<<<<< HEAD
        # 1. Battery SOC
        axes[0].plot(x_range, history["soc"], color='blue')
        axes[0].set_title("Battery SOC (Day 252)", fontsize=14)
        axes[0].set_ylabel("SOC (%)")
=======
        # 1. Battery SOC vs Time — detailed annotated chart
        soc_arr = np.array(history["soc"])
        lmp_arr = np.array(history["lmp"])
        hours   = np.arange(len(soc_arr))

        # Color fill: green when charging (SOC rising), red when discharging
        for i in range(1, len(hours)):
            color = '#27ae60' if soc_arr[i] >= soc_arr[i-1] else '#e74c3c'
            axes[0].fill_between([hours[i-1], hours[i]], [soc_arr[i-1], soc_arr[i]], alpha=0.35, color=color)

        axes[0].plot(hours, soc_arr, color='#2980b9', linewidth=1.2, label='SOC', zorder=3)

        # 20% minimum reserve threshold
        axes[0].axhline(0.20, color='#f39c12', linestyle='--', linewidth=1.0, alpha=0.8, label='20% Reserve')

        # Day boundary markers (every 24 hours)
        for day in range(1, len(soc_arr) // 24 + 1):
            axes[0].axvline(day * 24, color='white', linewidth=0.5, alpha=0.4, linestyle=':')

        # LMP overlay on secondary y-axis
        ax_soc_twin = axes[0].twinx()
        ax_soc_twin.plot(hours, lmp_arr, color='#f1c40f', linewidth=0.8, alpha=0.5, label='LMP')
        ax_soc_twin.set_ylabel('LMP ($/MWh)', color='#f1c40f', fontsize=9)
        ax_soc_twin.tick_params(axis='y', labelcolor='#f1c40f')

        axes[0].set_title('Battery State of Charge vs Time', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('SOC (0–1)', color='#2980b9')
        axes[0].set_xlabel('Hour of Episode')
        axes[0].set_ylim(0, 1.05)
        axes[0].tick_params(axis='y', labelcolor='#2980b9')

        # Combined legend
        lines_soc  = axes[0].get_lines() + ax_soc_twin.get_lines()
        labels_soc = [l.get_label() for l in lines_soc]
        axes[0].legend(lines_soc, labels_soc, loc='upper left', fontsize=8)
>>>>>>> e312b64 (initial BESS-RL commit)
        
        # 2. Energy Arbitrage - LMP on left axis, action on right axis
        axes[1].plot(x_range, history["lmp"], color='orange', label="LMP ($)")
        axes[1].set_title("Energy Arbitrage", fontsize=14)
        axes[1].set_ylabel("LMP ($/MWh)", color='orange')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(x_range, history["action_ea"], color='green', alpha=0.7, label="EA Action")
        ax1_twin.set_ylabel("Action [-1, 1]", color='green')
        ax1_twin.set_ylim(-1.2, 1.2)
        lines1 = axes[1].get_lines() + ax1_twin.get_lines()
        labels1 = [l.get_label() for l in lines1]
        axes[1].legend(lines1, labels1, loc="upper right")
        
        # 3. Frequency Regulation - FR Reward on left axis, action on right axis
        axes[2].plot(x_range, history["r_fr"], color='purple', label="FR Reward ($)")
        axes[2].set_title("Frequency Regulation", fontsize=14)
        axes[2].set_ylabel("FR Reward ($)", color='purple')
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(x_range, history["action_fr"], color='grey', alpha=0.8, label="FR Action")
        ax2_twin.set_ylabel("Action [-1, 1]", color='grey')
        ax2_twin.set_ylim(-1.2, 1.2)
        lines2 = axes[2].get_lines() + ax2_twin.get_lines()
        labels2 = [l.get_label() for l in lines2]
        axes[2].legend(lines2, labels2, loc="upper right")
        
        # 4. Peak Shaving
        axes[3].plot(x_range, history["baseline_load"], color='red', label="Baseline")
        axes[3].plot(x_range, history["net_load"], color='black', label="Net")
        axes[3].set_title("Peak Shaving", fontsize=14)
        axes[3].legend(loc="upper right")
        
        output_file = "bess_visualization.png"
        fig.savefig(output_file, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
        print(f"Visualization saved to {os.path.abspath(output_file)}")
        
    finally:
        print("Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--model-path", type=str, default=None, help="Path base name for saved agent models")
    args = parser.parse_args()
    
    visualize(args)
