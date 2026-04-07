import os
import torch
import argparse
from safetensors.torch import save_file

def export_to_safetensors(task_args):
    tasks = []
    
    _SCRIPT_DIR = os.path.dirname(__file__)
    
    if task_args == "all":
        tasks = ["easy", "medium", "hard"]
        output_file = os.path.join(_SCRIPT_DIR, "models", "bess_RL_master_bundle.safetensors")
    else:
        tasks = [task_args]
        output_file = os.path.join(_SCRIPT_DIR, "models", f"bess_{task_args}_bundle.safetensors")

    combined_state_dict = {}
    
    for task in tasks:
        base_name = os.path.join(_SCRIPT_DIR, "models", f"best_model_{task}")
        
        # SAC Model Components
        modules = {
            "actor": f"{base_name}_actor.pth",
            "critic": f"{base_name}_critic.pth",
            "alpha": f"{base_name}_alpha.pth"
        }

        # Check if the actor exists first
        if not os.path.exists(modules["actor"]):
            print(f"Warning: Could not find {modules['actor']}. Skipping '{task}' tier.")
            continue

        print(f"Packing '{task}' tier...")
        for prefix, file_path in modules.items():
            if os.path.exists(file_path):
                # Using weights_only=True for safety, though alpha is a single tensor
                try:
                    state_dict = torch.load(file_path, map_location="cpu", weights_only=False)
                    
                    if isinstance(state_dict, torch.Tensor):
                        # Alpha is often saved as a single tensor
                        combined_state_dict[f"{task}.{prefix}"] = state_dict
                    else:
                        for key, tensor in state_dict.items():
                            # Combine them with specific task and module prefixes
                            # Example: "easy.actor.l1.weight"
                            combined_state_dict[f"{task}.{prefix}.{key}"] = tensor
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")

    if not combined_state_dict:
        print("Error: No models were found to pack!")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_file(combined_state_dict, output_file)
    print(f"Success! Bundle saved to: {output_file}")
    print(f"You can safely upload this single '.safetensors' file to Hugging Face!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["easy", "medium", "hard", "all"], help="Specify a tier, or 'all' to pack them all into one master file.")
    args = parser.parse_args()
    
    export_to_safetensors(args.task)
