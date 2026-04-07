import os
import sys
import numpy as np
import time
import subprocess
import argparse
import json

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Canonical models directory (matches llm_evaluate.py and evaluate.py)
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

from openenv.client import OpenEnvClient
from agent.config import AgentConfig
from agent.replay_buffer import ReplayBuffer
from agent.actor_critic import SAC_Agent as TDD_ND_Agent # Alias for compatibility

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
        [sys.executable, "backend/main.py"], # Use our established main.py
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5) # Wait for server startup
    return server_process

def train(args):
    server_process = start_server()
    
    try:
        config = AgentConfig()
        client = OpenEnvClient(base_url="http://127.0.0.1:8000")
        
        agent = TDD_ND_Agent(config)
        
        load_path = args.load_model if args.load_model else os.path.join(_MODELS_DIR, f"best_model_{args.task}")
        
        best_reward = -1000000.0
        model_loaded_ok = False
        
        # Architecture mismatch check
        if os.path.exists(load_path + "_actor.pth"):
            try:
                print(f"Attempting to resume from: {load_path}")
                # Note: SAC architecture is different from TDD-ND. 
                # This will likely fail if loading an old TDD-ND checkpoint.
                agent.load(load_path)
                print("Successfully loaded model weights.")
                model_loaded_ok = True
            except Exception as e:
                print(f"Architecture Mismatch: {e}")
                print("Old TDD-ND weights incompatible with SAC. Starting fresh.")
                model_loaded_ok = False
        
        # Load metadata only if model loaded successfully (avoids stale TDD-ND rewards)
        task_meta_path = os.path.join(_MODELS_DIR, f"best_model_{args.task}_meta.json")
        if model_loaded_ok and os.path.exists(task_meta_path):
            with open(task_meta_path, "r") as f:
                meta = json.load(f)
                best_reward = meta.get("best_reward", -1000000.0)
                agent.total_it = meta.get("total_it", 0)
            print(f"Resumed best reward for task '{args.task}': {best_reward:.2f}")

        replay_buffer = ReplayBuffer(config.state_dim, config.action_dim, config.buffer_size)
        
        print(f"Starting SAC Agent Training on task: {args.task}")
        os.makedirs(_MODELS_DIR, exist_ok=True)
        
        total_steps = 0
        
        for ep in range(args.episodes):
            state = client.reset(seed=ep + int(time.time())%1000, task=args.task)
            ep_reward = 0
            ep_steps = 0
            done = False
            
            while not done:
                # SAC uses stochastic sampling during training
                if total_steps < config.exploration_steps and not os.path.exists(load_path + "_actor.pth"):
                    action = np.random.uniform(-config.max_action, config.max_action, size=config.action_dim)
                else:
                    action = agent.select_action(np.array(state), evaluate=False)
                
                next_state, reward, terminated, truncated, _ = client.step(action)
                done = terminated or truncated
                
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                
                if replay_buffer.size > config.batch_size:
                    agent.train(replay_buffer)
            
            # Print metrics
            alpha_val = agent.alpha.item() if hasattr(agent, 'alpha') else 0.0
            print(f"Ep {ep+1}/{args.episodes} | Reward: {ep_reward:.2f} | Steps: {ep_steps} | Alpha: {alpha_val:.4f}")
            
            # Save best model
            if ep_reward > best_reward:
                best_reward = ep_reward
                save_path = os.path.join(_MODELS_DIR, f"best_model_{args.task}")
                agent.save(save_path)
                with open(os.path.join(_MODELS_DIR, f"best_model_{args.task}_meta.json"), "w") as f:
                    json.dump({"best_reward": best_reward, "total_it": agent.total_it, "algorithm": "SAC"}, f)
                print(f"  --> New best model saved! Reward: {best_reward:.2f}")

    except Exception as outer_e:
        print(f"Training Error: {outer_e}")
    finally:
        if server_process:
            print("Shutting down local server...")
            server_process.terminate()
        else:
            print("Training session finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=50) # Increased default for SAC
    parser.add_argument("--load-model", type=str, default=None)
    args = parser.parse_args()
    
    train(args)
