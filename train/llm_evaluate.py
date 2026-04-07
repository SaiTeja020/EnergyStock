import os
import sys
import argparse
import json
from dotenv import load_dotenv

# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import evaluation logic
from train.evaluate import evaluate_model, score_model, OpenEnvClient
from backend.api.llm_evaluator import get_llm_analysis

def main():
    parser = argparse.ArgumentParser(description="BESS-RL LLM Evaluation CLI")
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"], help="Task complexity")
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation seeds")
    parser.add_argument("--model", type=str, default=None, help="Specific model name (defaults to best_model_<task>)")
    args = parser.parse_args()

    # Load API Key
    load_dotenv(os.path.join(_ROOT, ".env"))
    
    model_name = args.model if args.model else f"best_model_{args.task}"
    model_path = os.path.join(_ROOT, "train", "models", model_name)
    
    # Initialize Client
    client = OpenEnvClient(base_url="http://127.0.0.1:8000")
    
    try:
        # Check if server is up
        import requests
        requests.get("http://127.0.0.1:8000/api/health", timeout=2)
    except:
        print("Error: Backend server not detected at http://127.0.0.1:8000")
        sys.exit(1)

    eval_seeds = list(range(300, 300 + args.seeds))
    results = evaluate_model(client, model_path, args.task, eval_seeds)
    scores, overall = score_model(results, args.task)
    
    # Prepare data for LLM
    llm_input = {
        **results,
        "task": args.task,
        "model_name": model_name,
        "num_seeds": args.seeds,
        "scores": {**scores, "overall": overall}
    }

    # Call LLM Evaluator
    analysis = get_llm_analysis(llm_input)

    # OUTPUT: Only Score and Verdict as requested
    final_score = analysis.get('score', overall)
    verdict = analysis.get('verdict', 'N/A')
    
    # print(f"Overall Score (AI Evaluated): {final_score:.3f}")
    provider = analysis.get('provider', 'AI')
    print(f"Overall Score ({provider} Evaluated): {final_score:.3f}")
    print(f"AI Verdict: {verdict}")

if __name__ == "__main__":
    main()
