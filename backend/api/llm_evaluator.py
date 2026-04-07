"""
LLM Evaluator – analyzes BESS-RL evaluation results using Gemini or Hugging Face.
Includes a heuristic fallback for when API quotas are exceeded.
"""
import os
import json
import requests

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2-9b-it")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "GEMINI").upper()

_FALLBACK_BASE = {
    "available": False,
    "verdict": "N/A",
    "score": 0.0,
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "recommendations": [],
    "confidence": "N/A",
    "detailed_analysis": "",
    "error": None,
    "is_heuristic": False,
    "provider": "UNKNOWN"
}

TASK_DESCRIPTIONS = {
    "easy":   "Energy Arbitrage only (buy cheap, sell expensive)",
    "medium": "Energy Arbitrage + Frequency Regulation",
    "hard":   "Energy Arbitrage + Frequency Regulation + Peak Shaving",
}

def _get_heuristic_analysis(data: dict) -> dict:
    """Provides a rule-based assessment when LLMs are unavailable."""
    scores = data.get("scores", {})
    overall = scores.get("overall", 0)
    task = data.get("task", "hard")
    
    # Determine Verdict
    if overall >= 0.85: verdict = "Excellent"
    elif overall >= 0.70: verdict = "Good"
    elif overall >= 0.50: verdict = "Needs Improvement"
    else: verdict = "Poor"
    
    strengths, weaknesses, recommendations = [], [], []
    
    # Analysis logic
    if scores.get("reward", 0) > 0.8: strengths.append("Strong reward optimization")
    else: weaknesses.append("Sub-optimal profit generation")
    
    if scores.get("soc_readiness", 0) > 0.8: strengths.append("Excellent peak-hour SOC management")
    else: recommendations.append("Improve SOC buffer management before evening peaks")
    
    if scores.get("ps_adherence", 0) < 0.9 and task == "hard":
        weaknesses.append("Frequent peak shaving violations")
        recommendations.append("Increase penalty factor for grid load violations in reward function")
        
    if scores.get("cycle_discipline", 0) < 0.6:
        weaknesses.append("Aggressive battery cycling")
        recommendations.append("Increase degradation cost coefficient to preserve battery health")

    # Calculate an independent heuristic score (weighted average of components)
    # This makes the heuristic score distinct from the simulator's engine score
    h_score = (
        scores.get("reward", 0) * 0.4 +
        scores.get("soc_readiness", 0) * 0.2 +
        scores.get("ps_adherence", 0) * 0.3 +
        scores.get("cycle_discipline", 0) * 0.1
    )

    return {
        **_FALLBACK_BASE,
        "available": True,
        "is_heuristic": True,
        "provider": "LOCAL",
        "verdict": verdict,
        "score": round(h_score, 3),
        "summary": f"This is a heuristic assessment based on numerical performance metrics. The agent shows a {verdict.lower()} grasp of the {task} task.",
        "strengths": strengths or ["Stable baseline performance"],
        "weaknesses": weaknesses or ["Minor edge-case inefficiencies"],
        "recommendations": recommendations or ["Continue training with randomized seeding"],
        "confidence": "Medium (Rule-based)",
        "detailed_analysis": "This analysis was generated locally because the LLM API is currently unavailable. It focuses on the core operational KPIs: Profitability, SOC management, and Constraint Adherence."
    }

def _build_prompt(data: dict) -> str:
    task = data.get("task", "unknown")
    scores = data.get("scores", {})
    return f"""You are an expert in reinforcement learning for grid-scale energy storage systems.
Analyse the following BESS-RL (Battery Energy Storage System – Reinforcement Learning) agent
evaluation results. The agent uses a Soft Actor-Critic (SAC) stochastic architecture.
Return a thorough, actionable JSON assessment.

== EVALUATION CONTEXT ==
Task        : {task} — {TASK_DESCRIPTIONS.get(task, '')}
Model       : {data.get('model_name', 'unknown')}
Seeds used  : {data.get('num_seeds', '?')} (unseen, held-out)

== RAW METRICS ==
Mean Reward          : {data.get('reward_mean', 0):.1f}
SOC at Peak Hours    : {data.get('soc_at_peak_mean', 0):.1%}   (target > 70 %)
Peak Shaving Viol.   : {data.get('peak_violation_pct', 0):.1f}%  (target < 5 %)
Avg Cycles / Episode : {data.get('avg_cycles_per_ep', 0):.1f}
Arbitrage Accuracy   : {data.get('arb_accuracy_pct', 0):.1f}%

== NORMALISED SCORE BREAKDOWN (0–1) ==
Reward Score         : {scores.get('reward', 0):.3f}
SOC Readiness        : {scores.get('soc_readiness', 0):.3f}
Peak Shaving Adh.    : {scores.get('ps_adherence', 0):.3f}
Cycle Discipline     : {scores.get('cycle_discipline', 0):.3f}

== INSTRUCTIONS ==
Return ONLY a single valid JSON object with exactly these keys:
{{
  "verdict": "Excellent|Good|Needs Improvement|Poor",
  "score": <float between 0.0 and 1.0 representing overall system performance>,
  "summary": "<2-3 sentence executive summary>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "recommendations": ["<rec 1>", "<rec 2>"],
  "confidence": "High|Medium|Low",
  "detailed_analysis": "<3 paragraph technical analysis>"
}}
"""

def _get_gemini_analysis(data: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not set")
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = _build_prompt(data)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            # response_mime_type="application/json",  # Some regions don't support JSON mode yet
        ),
    )
    # Extract JSON string safely
    text = response.text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
        
    return {**json.loads(text.strip()), "available": True, "provider": "GEMINI"}

def _get_hf_analysis(data: dict) -> dict:
    api_token = os.getenv("HF_TOKEN")
    if not api_token: raise ValueError("HF_TOKEN not set")
    
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}"}
    prompt = _build_prompt(data) + "\nJSON Output:"
    
    response = requests.post(api_url, headers=headers, json={
        "inputs": prompt,
        "parameters": {"max_new_tokens": 1000, "return_full_text": False}
    })
    
    # Extract JSON string from response (some models include text wrap)
    text = response.json()[0]['generated_text']
    try:
        # Simple extraction if model outputs markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        return {**json.loads(text.strip()), "available": True, "provider": "HF"}
    except:
        # Fallback if raw JSON extraction fails
        raise ValueError("Failed to parse JSON from Hugging Face model response")

def get_llm_analysis(data: dict) -> dict:
    # Try Hugging Face first (preferred by user)
    try:
        if os.getenv("HF_TOKEN"):
            return _get_hf_analysis(data)
    except Exception as hf_err:
        pass

    # Try Gemini as second choice
    try:
        if os.getenv("GEMINI_API_KEY"):
            return _get_gemini_analysis(data)
    except Exception as gem_err:
        pass

    # Final fallback: Heuristic
    return _get_heuristic_analysis(data)
