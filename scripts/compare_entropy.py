import os
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Paths


finetune_dir = "saves/finetune"
full_model_path = "/home/cnz/project/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_full"
retain99_model_path = "/home/cnz/project/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain99"
forget_dataset_path = "/home/cnz/.cache/huggingface/hub/datasets--locuslab--TOFU/snapshots/324592d84ae4f482ac7249b9285c2ecdb53e3a68"
forget_files = ["forget01.json", "forget01_perturbed.json"]
origin_model_path = "/home/cnz/.cache/huggingface/hub/Llama-3___2-1B-Instruct"

# Load dataset (supports JSONL and JSON list)
def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    # JSON list
    if content[0] == "[":
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except JSONDecodeError:
            pass
    # JSON Lines
    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except JSONDecodeError:
            # Skip malformed lines but continue
            continue
    return records

# Load model and tokenizer
def load_model(model_path: str):
    # Use a shared tokenizer (origin) for consistent tokenization across finetunes
    tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
    # Prefer lower precision on GPU to save memory
    kwargs = {}
    if torch.cuda.is_available():
        # Use float16 by default for compatibility; bf16 only on supported GPUs
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model

# Calculate entropy
def calculate_entropy(probabilities):
    return entropy(probabilities, base=2)

def _pick_first_str(sample: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if k == "perturbed_answer" and isinstance(v, list) and v:
            # Use first candidate if answers are provided as a list
            first = v[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None

def _build_chat_prompt(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> Optional[str]:
    q = _pick_first_str(sample, [
        "question",
        "paraphrased_question",
        "input",
        "prompt",
        "instruction",
        "text",
    ])
    a = _pick_first_str(sample, [
        "answer",
        "paraphrased_answer",
        "perturbed_answer",
    ])
    if not q and not a:
        return None
    messages = []
    if q:
        messages.append({"role": "user", "content": q})
    if a:
        messages.append({"role": "assistant", "content": a})

    # Build chat-formatted prompt string; no extra generation tag to measure next-token after provided content
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Fallback if tokenizer lacks chat template
    joined = "\n\n".join([f"User: {q}" if q else "", f"Assistant: {a}" if a else ""]).strip()
    return joined if joined else None

# Build full chat text and locate answer token start index
def _build_full_and_answer_start(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> Optional[Dict[str, Any]]:
    q = _pick_first_str(sample, [
        "question",
        "paraphrased_question",
        "input",
        "prompt",
        "instruction",
        "text",
    ])
    a = _pick_first_str(sample, [
        "answer",
        "paraphrased_answer",
        "perturbed_answer",
    ])
    if not a:
        return None  # no answer -> skip for answer-only entropy
    messages = []
    if q:
        messages.append({"role": "user", "content": q})
    messages.append({"role": "assistant", "content": a})

    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if q:
            # user-only with generation prompt puts the assistant prefix; answer starts right after
            pre_text = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            ans_start = int(pre_ids.shape[0])
        else:
            # No user turn; find where assistant content begins in rendered full_text
            pos = full_text.find(a)
            if pos < 0:
                ans_start = 0
            else:
                pre_text = full_text[:pos]
                pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                ans_start = int(pre_ids.shape[0])
        return {"text": full_text, "ans_start": ans_start}
    # Fallback without chat template
    if q:
        pre_text = f"User: {q}\n\nAssistant:"
        full_text = f"{pre_text} {a}"
        pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(pre_ids.shape[0])
    else:
        full_text = f"Assistant: {a}"
        pos = full_text.find(a)
        pre_text = full_text[:pos] if pos >= 0 else ""
        pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(pre_ids.shape[0])
    return {"text": full_text, "ans_start": ans_start}

def _build_chat_from_strings(q: Optional[str], a: Optional[str], tokenizer: AutoTokenizer) -> Optional[Dict[str, Any]]:
    if not a:
        return None
    messages = []
    if q:
        messages.append({"role": "user", "content": q})
    messages.append({"role": "assistant", "content": a})

    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if q:
            pre_text = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            ans_start = int(pre_ids.shape[0])
        else:
            pos = full_text.find(a)
            if pos < 0:
                ans_start = 0
            else:
                pre_text = full_text[:pos]
                pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                ans_start = int(pre_ids.shape[0])
        return {"text": full_text, "ans_start": ans_start}
    # Fallback
    if q:
        pre_text = f"User: {q}\n\nAssistant:"
        full_text = f"{pre_text} {a}"
        pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(pre_ids.shape[0])
    else:
        full_text = f"Assistant: {a}"
        pos = full_text.find(a)
        pre_text = full_text[:pos] if pos >= 0 else ""
        pre_ids = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ans_start = int(pre_ids.shape[0])
    return {"text": full_text, "ans_start": ans_start}

# Evaluate model
def evaluate_model(model, tokenizer, dataset: List[Dict[str, Any]], max_samples: Optional[int] = None, max_length: int = 512) -> float:
    entropies: List[float] = []
    device = next(model.parameters()).device
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        built = _build_full_and_answer_start(sample, tokenizer)
        if not built:
            continue
        try:
            text = built["text"]
            ans_start = built["ans_start"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                # We want entropy over positions predicting answer tokens only
                seq_len = logits.shape[1]
                # positions 0..seq_len-2 predict tokens 1..seq_len-1
                start_pos = max(ans_start - 1, 0)
                end_pos = seq_len - 2
                if end_pos >= start_pos:
                    probs = torch.softmax(logits[:, start_pos:end_pos + 1, :], dim=-1)[0]  # (span, V)
                    # compute entropy per position and average
                    ent_vals = []
                    for row in probs:
                        ent_vals.append(float(calculate_entropy(row.detach().cpu().numpy())))
                    if ent_vals:
                        entropies.append(float(sum(ent_vals) / len(ent_vals)))
        except Exception:
            # Skip any problematic sample to keep the run going
            continue
    return float(sum(entropies) / max(len(entropies), 1))

def evaluate_model_variants(model, tokenizer, dataset: List[Dict[str, Any]], max_samples: Optional[int] = None, max_length: int = 512) -> Dict[str, float]:
    """Compute answer-only entropy for combinations of (q,a):
    - orig_q_orig_a, para_q_orig_a, orig_q_para_a, para_q_para_a
    Returns average entropy per variant over the dataset.
    """
    device = next(model.parameters()).device
    entropies: Dict[str, List[float]] = {k: [] for k in [
        "orig_q_orig_a", "para_q_orig_a", "orig_q_para_a", "para_q_para_a"
    ]}
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        q_orig = _pick_first_str(sample, ["question", "input", "prompt", "instruction", "text"])  # prefer original
        q_para = _pick_first_str(sample, ["paraphrased_question"]) or None
        a_orig = _pick_first_str(sample, ["answer"])  # require
        a_para = _pick_first_str(sample, ["paraphrased_answer"]) or None

        variants = {
            "orig_q_orig_a": (q_orig, a_orig),
            "para_q_orig_a": (q_para, a_orig),
            "orig_q_para_a": (q_orig, a_para),
            "para_q_para_a": (q_para, a_para),
        }
        for name, (q, a) in variants.items():
            if not a:  # need an answer to compute answer-span entropy
                continue
            try:
                built = _build_chat_from_strings(q, a, tokenizer)
                if not built:
                    continue
                text = built["text"]
                ans_start = built["ans_start"]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    seq_len = logits.shape[1]
                    start_pos = max(ans_start - 1, 0)
                    end_pos = seq_len - 2
                    if end_pos >= start_pos:
                        probs = torch.softmax(logits[:, start_pos:end_pos + 1, :], dim=-1)[0]
                        ent_vals = []
                        for row in probs:
                            ent_vals.append(float(calculate_entropy(row.detach().cpu().numpy())))
                        if ent_vals:
                            entropies[name].append(float(sum(ent_vals) / len(ent_vals)))
            except Exception:
                continue

    return {k: (float(sum(v) / len(v)) if len(v) > 0 else float('nan')) for k, v in entropies.items()}

def plot_results(results):
    """Plots the entropy results and saves the plot."""
    dataset_names = list(results.keys())
    model_names = ['origin_model', 'full_model', 'retain99_model']
    entropies = {model: [results[ds][f'{model}_entropy'] for ds in dataset_names] for model in model_names}

    x = np.arange(len(dataset_names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in entropies.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_ylabel('Average Entropy')
    ax.set_title('Model Output Entropy on Forget Sets')
    ax.set_xticks(x + width, dataset_names)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, max(max(e) for e in entropies.values()) * 1.2) # Adjust y-limit for better visualization

    plt.savefig("scripts/entropy_comparison.png")
    print("Plot saved to scripts/entropy_comparison.png")

def plot_variant_results(dataset_name: str, model_to_variant: Dict[str, Dict[str, float]]):
    variants = ["orig_q_orig_a", "para_q_orig_a", "orig_q_para_a", "para_q_para_a"]
    models = ["origin_model", "full_model", "retain99_model"]
    x = np.arange(len(variants))
    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    for i, m in enumerate(models):
        vals = [model_to_variant.get(m, {}).get(v, np.nan) for v in variants]
        rects = ax.bar(x + i*width, vals, width, label=m)
        ax.bar_label(rects, padding=3, fmt='%.2f')
    ax.set_ylabel('Avg Answer-only Entropy')
    ax.set_title(f'Variants entropy on {dataset_name}')
    ax.set_xticks(x + width, variants)
    ax.legend(loc='upper left', ncols=3)
    # y-limit guard
    finite_vals = [v for m in models for v in [model_to_variant.get(m, {}).get(u, np.nan) for u in variants] if np.isfinite(v)]
    if finite_vals:
        ax.set_ylim(0, max(finite_vals) * 1.2)
    out_path = f"scripts/entropy_variants_{dataset_name.replace('.json','')}.png"
    plt.savefig(out_path)
    print(f"Variant plot saved to {out_path}")


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare output entropy across models on TOFU forget sets.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per dataset for quick runs")
    parser.add_argument("--max-length", type=int, default=512, help="Max input token length")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Override dataset files to evaluate")
    args = parser.parse_args()
    # Define models to evaluate
    model_paths = {
        "origin_model": origin_model_path,
        "full_model": full_model_path,
        "retain99_model": retain99_model_path,
    }

    # Load datasets
    # Default to perturbed-only unless user overrides
    default_files = [f for f in forget_files if "perturbed" in f]
    use_files = args.datasets if args.datasets else (default_files if default_files else forget_files)
    datasets = {file: load_dataset(os.path.join(forget_dataset_path, file)) for file in use_files}

    # Initialize results structure
    results = {name: {} for name in datasets.keys()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluate models one by one to save memory
    for model_name, model_path in model_paths.items():
        print(f"Loading and evaluating model: {model_name}")
        
        # Load model and tokenizer
        tokenizer, model = load_model(model_path)
        model.to(device)

        for dataset_name, dataset in datasets.items():
            print(f"  - on dataset: {dataset_name}")
            if "perturbed" in dataset_name:
                # Evaluate four variants per sample and average
                variant_avgs = evaluate_model_variants(model, tokenizer, dataset, max_samples=args.max_samples, max_length=args.max_length)
                # Store per model
                results.setdefault(dataset_name, {})[f'{model_name}_variants'] = variant_avgs
                # Also store a simple baseline using original (orig_q_orig_a)
                results[dataset_name][f'{model_name}_entropy'] = variant_avgs.get("orig_q_orig_a", float('nan'))
            else:
                entropy_val = evaluate_model(model, tokenizer, dataset, max_samples=args.max_samples, max_length=args.max_length)
                results[dataset_name][f'{model_name}_entropy'] = entropy_val
        
        # Clear memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Finished evaluating {model_name} and cleared memory.")


    # Print results
    for dataset_name, result in results.items():
        print(f"Dataset: {dataset_name}")
        print(f"  Origin model entropy: {result['origin_model_entropy']:.4f}")
        print(f"  Full model entropy: {result['full_model_entropy']:.4f}")
        print(f"  Retain99 model entropy: {result['retain99_model_entropy']:.4f}")

    # Plot overall entropy comparison if any non-variant entries exist
    has_simple = any(all(k.endswith("_entropy") for k in v.keys()) for v in results.values())
    if has_simple:
        plot_results({k: v for k, v in results.items() if all(x.endswith('_entropy') for x in v.keys())})

    # Plot variant results for perturbed datasets
    for dataset_name, model_map in results.items():
        if any(name.endswith('_variants') for name in model_map.keys()):
            model_to_variant = {}
            for m in ["origin_model", "full_model", "retain99_model"]:
                key = f"{m}_variants"
                if key in model_map:
                    model_to_variant[m] = model_map[key]
            if model_to_variant:
                plot_variant_results(dataset_name, model_to_variant)
