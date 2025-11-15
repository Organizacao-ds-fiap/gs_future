# %%
# Creating a synthetic dataset for the "LLM Matchmaker" project:
# - 1000 rows
# - 5 candidate models
# - Target: best_model (classification)
# %%
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

N = 1000

# Candidate models
models = [
    "GPT-4o",
    "Gemini",
    "Claude-2",
    "Llama-3-70B",
    "Deepseek"
]

# Possible categorical values
task_types = ["classification", "summarization", "generation", "reasoning", "extraction"]
domains = ["general", "legal", "medical", "finance", "ecommerce", "technical"]
languages = ["pt", "en", "multi"]
hardware_opts = ["cpu", "consumer_gpu", "pro_gpu", "edge"]
privacy_opts = ["cloud", "local", "hybrid"]  # local => prefers local models
hallucination_levels = ["low", "medium", "high"]
determinism_opts = [0, 1]  # 1 => determinism needed
temperature_prefs = ["low", "medium", "high"]

# Model capabilities (meta) — simplified and hypothetical
model_meta = {
    "GPT-4o": {
        "params_b": 0.05,    # relative scale
        "strengths": ["reasoning", "summarization"],
        "languages": ["en","pt","multi"],
        "latency_base_ms": 200,
        "cost_per_1k": 6.0,
        "offline": False
    },
    "Gemini": {
        "params_b": 0.13,
        "strengths": ["summarization","generation"],
        "languages": ["en","pt","multi"],
        "latency_base_ms": 240,
        "cost_per_1k": 15.0,
        "offline": False
    },
    "Claude-2": {
        "params_b": 0.08,
        "strengths": ["reasoning","classification"],
        "languages": ["en","multi"],
        "latency_base_ms": 220,
        "cost_per_1k": 8.0,
        "offline": False
    },
    "Llama-3-70B": {
        "params_b": 70,
        "strengths": ["extraction","classification"],
        "languages": ["en","pt","multi"],
        "latency_base_ms": 180,
        "cost_per_1k": 0.5,
        "offline": True
    },
    "Deepseek": {
        "params_b": 7,
        "strengths": ["generation","extraction"],
        "languages": ["en","pt"],
        "latency_base_ms": 90,
        "cost_per_1k": 0.2,
        "offline": True
    }
}

rows = []
for i in range(N):
    task_type = random.choice(task_types)
    domain = random.choice(domains)
    input_language = random.choice(languages)
    # avg_input_length and n_samples removed
    privacy_requirement = random.choices(["cloud","local","hybrid"], weights=[0.6,0.25,0.15])[0]
    # max_latency_ms and max_cost_per_1k removed
    hardware_available = random.choices(hardware_opts, weights=[0.25,0.45,0.2,0.1])[0]
    hallucination_tolerance = random.choice(hallucination_levels)
    determinism_needed = random.choice(determinism_opts)
    temperature_pref = random.choice(temperature_prefs)

    # Derived features
    input_type = "text"
    structured_input = 0  # mostly text tasks here
    output_style = random.choice(["creative","formal","precise","factual"])

    # Now score each model according to heuristics and pick best
    scores = {}
    for m in models:
        meta = model_meta[m]
        score = 0.0

        # Task match: if model lists task_type as strength
        if task_type in meta["strengths"]:
            score += 2.0
        # Domain sensitivity (simple): legal/medical prefer Claude/GPT-4 for reasoning/classification
        if domain in ["legal","medical","finance"]:
            if m in ["Gemini","Claude-2"]:
                score += 1.5
            if m in ["Llama-3-70B","Deepseek"]:
                score += 0.5
        # Language support
        if input_language in meta["languages"]:
            score += 1.0
        else:
            score -= 0.5

        # Latency and cost constraints removed

        # Privacy
        if privacy_requirement == "local":
            if meta["offline"]:
                score += 2.0
            else:
                score -= 2.0
        elif privacy_requirement == "hybrid":
            if meta["offline"]:
                score += 0.8

        # Hardware: if edge or cpu, favor smaller models
        if hardware_available == "edge":
            if m in ["Deepseek"]:
                score += 1.5
            elif m in ["Llama-3-70B","Gemini","Claude-2","GPT-4o"]:
                score -= 1.0
        if hardware_available == "cpu":
            if m in ["Deepseek","Llama-3-70B"]:
                score += 0.8
            else:
                score -= 0.8

        # Hallucination tolerance / determinism
        if hallucination_tolerance == "low":
            if m in ["Gemini","Claude-2"]:
                score += 1.0
            else:
                score -= 0.5
        if determinism_needed == 1:
            if temperature_pref == "low":
                score += 0.3

        # favor models with matching strengths to output style
        if output_style in ["creative"] and m in ["GPT-4o","Deepseek"]:
            score += 0.7
        if output_style in ["precise","factual"] and m in ["Claude-2","Gemini"]:
            score += 0.7

        # slight bias for larger models on very large sample sizes for fine-tuning scenarios removed

        # add small random noise to avoid ties
        score += np.random.normal(0, 0.2)
        scores[m] = score

    # Choose best model (argmax)
    best_model = max(scores.items(), key=lambda x: x[1])[0]

    row = {
        "task_type": task_type,
        "domain": domain,
        "input_language": input_language,
        "privacy_requirement": privacy_requirement,
        "hardware_available": hardware_available,
        "hallucination_tolerance": hallucination_tolerance,
        "determinism_needed": determinism_needed,
        "temperature_pref": temperature_pref,
        "output_style": output_style,
        "best_model": best_model
    }
    # also include model score columns (optional) for transparency
    for m in models:
        row[f"score_{m}"] = round(scores[m], 3)
    rows.append(row)

df = pd.DataFrame(rows)

# Save CSV
out_path = Path("data/llm_matchmaker_dataset_1000.csv")
df.to_csv(out_path, index=False)

print(f"Synthetic dataset saved to: {out_path.resolve()}")
print(df["best_model"].value_counts())
df.sample(5, random_state=42)

