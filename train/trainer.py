import os
import sys
import numpy as np
import time
import subprocess
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bess_rl.openenv.client import OpenEnvClient
from bess_rl.agent.config import AgentConfig
from bess_rl.agent.replay_buffer import ReplayBuffer
from bess_rl.agent.actor_critic import TDD_ND_Agent

def start_server():
    print("Starting OpenEnv Server on port 8000...")
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    server_process = subprocess.Popen(
        ["uvicorn", "bess_rl.openenv.server.app:app", "--port", "8000"],
        env=env_vars,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3) # Wait for server startup
    return server_process

def train(args):
    server_process = start_server()
    
    try:
        config = AgentConfig()
        client = OpenEnvClient(base_url="http://127.0.0.1:8000")
        
        agent = TDD_ND_Agent(config)
        
        load_path = args.load_model if args.load_model else f"models/best_model_{args.task}"
        
        best_reward = -np.inf
        
        if os.path.exists(load_path + "_actor.pth"):
            try:
                print(f"Resuming training from checkpoint: {load_path}")
                agent.load(load_path)
            except Exception as e:
                print(f"Failed to load checkpoint (dimension mismatch due to state update): {e}")
                print("Starting from scratch with randomly initialized weights.")
        else:
            print(f"No checkpoint found at {load_path}. Starting from scratch with randomly initialized weights.")
        
        # Always read best_reward from the TARGET task's meta, not the source checkpoint's meta
        import json
        task_meta_path = f"models/best_model_{args.task}_meta.json"
        if os.path.exists(task_meta_path):
            with open(task_meta_path, "r") as f:
                meta = json.load(f)
                best_reward = meta.get("best_reward", -np.inf)
                agent.total_it = meta.get("total_it", agent.total_it)
            print(f"Recovered previous best reward for task '{args.task}': {best_reward:.2f}")
        else:
            print(f"No prior '{args.task}' task metadata found. Starting best_reward tracking from -inf.")
            
        replay_buffer = ReplayBuffer(config.state_dim, config.action_dim, config.buffer_size)
        
        print(f"Starting TDD-ND Agent Training on task: {args.task}")
        os.makedirs("models", exist_ok=True)
        
        total_steps = 0
        
        # If we loaded a pretrained model, skip straight to min noise
        # (the model has already explored — re-exploring just poisons the buffer)
        model_was_loaded = os.path.exists(load_path + "_actor.pth")
        if model_was_loaded:
            total_steps = config.exploration_decay_steps
            print(f"Pretrained model detected — starting noise at minimum ({config.exploration_noise_end}) to avoid buffer poisoning.")
        
        for ep in range(args.episodes):
            state = client.reset(seed=ep, task=args.task)
            ep_reward = 0
            ep_steps = 0
            done = False
            
            # ND (Noise Decay) - based on TOTAL STEPS, not episodes, to correctly reach minimum
            noise = max(config.exploration_noise_end,
                        config.exploration_noise_start - total_steps * (config.exploration_noise_start / config.exploration_decay_steps))
            
            while not done:
                # Select action with noise decay
                action = (
                    agent.select_action(np.array(state))
                    + np.random.normal(0, config.max_action * noise, size=config.action_dim)
                ).clip(-config.max_action, config.max_action)
                
                next_state, reward, terminated, truncated, _ = client.step(action)
                done = terminated or truncated
                
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                
                if replay_buffer.size > config.batch_size:
                    agent.train(replay_buffer)
                    
            print(f"Episode {ep+1}/{args.episodes} - Reward: {ep_reward:.2f} - Steps: {ep_steps}")
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(f"models/best_model_{args.task}")
                
                import json
                with open(f"models/best_model_{args.task}_meta.json", "w") as f:
                    json.dump({"best_reward": best_reward, "total_it": agent.total_it}, f)
                    
                print(f"New best model saved with reward: {best_reward:.2f}")
            
            
    finally:
        print("Shutting down server...")
        server_process.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--load-model", type=str, default=None, help="Path base to load model weights from (e.g. models/best_model_easy)")
    args = parser.parse_args()
    
    train(args)
