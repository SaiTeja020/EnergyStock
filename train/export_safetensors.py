import os
import torch
import argparse
from safetensors.torch import save_file

def export_to_safetensors(task_args):
    tasks = []
    
    if task_args == "all":
        tasks = ["easy", "medium", "hard"]
        output_file = "models/bess_RL_master_bundle.safetensors"
    else:
        tasks = [task_args]
        output_file = f"models/bess_{task_args}_bundle.safetensors"

    combined_state_dict = {}
    
    for task in tasks:
        base_name = f"models/best_model_{task}"
        modules = {
            "actor": f"{base_name}_actor.pth",
            "critic": f"{base_name}_critic.pth",
            "actor_target": f"{base_name}_actor_target.pth",
            "critic_target": f"{base_name}_critic_target.pth"
        }

        # Check if the actor exists first
        if not os.path.exists(modules["actor"]):
            print(f"Warning: Could not find {modules['actor']}. Skipping '{task}' tier.")
            continue

        for prefix, file_path in modules.items():
            if os.path.exists(file_path):
                state_dict = torch.load(file_path, weights_only=True)
                for key, tensor in state_dict.items():
                    # Combine them with specific task and module prefixes to prevent key collision
                    # Example: "easy.actor.l1.weight"
                    combined_state_dict[f"{task}.{prefix}.{key}"] = tensor

    if not combined_state_dict:
        print("Error: No models were found to pack!")
        return

    save_file(combined_state_dict, output_file)
    print(f"Success! Bundle saved to: {output_file}")
    print(f"You can safely upload this single '.safetensors' file to Hugging Face!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["easy", "medium", "hard", "all"], help="Specify a tier, or 'all' to pack them all into one master file.")
    args = parser.parse_args()
    
    export_to_safetensors(args.task)
