#!/usr/bin/env python3
"""
Analyze GSM-8K Relearn Attack Results

This script compares model performance across different stages of the relearn attack:
- Stage 1: Fine-tuned on GSM-8K
- Stage 2: After unlearning
- Stage 3: After relearn attack

Usage:
    python scripts/analyze_gsm8k_results.py --model Llama-3.2-1B-Instruct --method GradientAscent
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys


def load_summary(path: str) -> Optional[Dict]:
    """Load summary JSON file"""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def format_accuracy(acc: float) -> str:
    """Format accuracy as percentage"""
    return f"{acc * 100:.2f}%"


def analyze_model_results(model: str, method: str, base_dir: str = "saves/eval"):
    """Analyze results for a specific model and unlearn method"""
    print(f"\n{'='*80}")
    print(f"Analysis for Model: {model} | Unlearn Method: {method}")
    print(f"{'='*80}\n")
    
    # Define paths for different stages
    stage1_path = os.path.join(base_dir, f"gsm8k_{model}_stage1", "GSM8K_SUMMARY.json")
    stage2_path = os.path.join(base_dir, f"gsm8k_{model}_{method}_stage2_unlearn", "GSM8K_SUMMARY.json")
    stage3_path = os.path.join(base_dir, f"gsm8k_{model}_{method}_stage3_relearn", "GSM8K_SUMMARY.json")
    
    # Load summaries
    stage1 = load_summary(stage1_path)
    stage2 = load_summary(stage2_path)
    stage3 = load_summary(stage3_path)
    
    # Check if we have any results
    if not any([stage1, stage2, stage3]):
        print("No results found for this configuration.")
        return
    
    # Extract metrics
    results = []
    
    if stage1:
        results.append({
            "stage": "Stage 1: Fine-tuned",
            "accuracy": stage1.get("gsm8k_accuracy", 0),
            "correct": stage1.get("gsm8k_correct", 0),
            "total": stage1.get("gsm8k_total", 0),
        })
    
    if stage2:
        results.append({
            "stage": "Stage 2: After Unlearning",
            "accuracy": stage2.get("gsm8k_accuracy", 0),
            "correct": stage2.get("gsm8k_correct", 0),
            "total": stage2.get("gsm8k_total", 0),
        })
    
    if stage3:
        results.append({
            "stage": "Stage 3: After Relearn Attack",
            "accuracy": stage3.get("gsm8k_accuracy", 0),
            "correct": stage3.get("gsm8k_correct", 0),
            "total": stage3.get("gsm8k_total", 0),
        })
    
    # Print results table
    print(f"{'Stage':<35} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 62)
    
    for result in results:
        accuracy_str = format_accuracy(result["accuracy"])
        correct_total = f"{result['correct']}/{result['total']}"
        print(f"{result['stage']:<35} {accuracy_str:<12} {correct_total:<15}")
    
    # Calculate and print changes
    print(f"\n{'Impact Analysis':^62}")
    print("-" * 62)
    
    if stage1 and stage2:
        unlearn_impact = stage2["accuracy"] - stage1["accuracy"]
        print(f"Unlearning Impact:          {unlearn_impact:+.4f} ({format_accuracy(abs(unlearn_impact))} {'decrease' if unlearn_impact < 0 else 'increase'})")
    
    if stage2 and stage3:
        relearn_recovery = stage3["accuracy"] - stage2["accuracy"]
        print(f"Relearn Recovery:           {relearn_recovery:+.4f} ({format_accuracy(abs(relearn_recovery))} {'increase' if relearn_recovery > 0 else 'decrease'})")
    
    if stage1 and stage3:
        net_change = stage3["accuracy"] - stage1["accuracy"]
        recovery_rate = (stage3["accuracy"] - stage2["accuracy"]) / (stage1["accuracy"] - stage2["accuracy"]) * 100 if stage1 and stage2 else 0
        print(f"Net Change (Stage 3 vs 1):  {net_change:+.4f} ({format_accuracy(abs(net_change))} {'decrease' if net_change < 0 else 'increase'})")
        if stage2:
            print(f"Recovery Rate:              {recovery_rate:.2f}% of lost performance recovered")
    
    print(f"\n{'='*80}\n")


def compare_methods(model: str, methods: List[str], base_dir: str = "saves/eval"):
    """Compare different unlearn methods for a single model"""
    print(f"\n{'='*80}")
    print(f"Comparing Unlearn Methods for Model: {model}")
    print(f"{'='*80}\n")
    
    # Load stage 1 (baseline)
    stage1_path = os.path.join(base_dir, f"gsm8k_{model}_stage1", "GSM8K_SUMMARY.json")
    stage1 = load_summary(stage1_path)
    
    if stage1:
        print(f"Baseline (Stage 1 Fine-tuned): {format_accuracy(stage1['gsm8k_accuracy'])}\n")
    
    # Compare each method
    print(f"{'Method':<25} {'After Unlearn':<15} {'After Relearn':<15} {'Recovery Rate':<15}")
    print("-" * 70)
    
    for method in methods:
        stage2_path = os.path.join(base_dir, f"gsm8k_{model}_{method}_stage2_unlearn", "GSM8K_SUMMARY.json")
        stage3_path = os.path.join(base_dir, f"gsm8k_{model}_{method}_stage3_relearn", "GSM8K_SUMMARY.json")
        
        stage2 = load_summary(stage2_path)
        stage3 = load_summary(stage3_path)
        
        if stage2 and stage3 and stage1:
            stage2_acc = format_accuracy(stage2["gsm8k_accuracy"])
            stage3_acc = format_accuracy(stage3["gsm8k_accuracy"])
            
            # Calculate recovery rate
            lost = stage1["gsm8k_accuracy"] - stage2["gsm8k_accuracy"]
            recovered = stage3["gsm8k_accuracy"] - stage2["gsm8k_accuracy"]
            recovery_rate = (recovered / lost * 100) if lost > 0 else 0.0
            
            print(f"{method:<25} {stage2_acc:<15} {stage3_acc:<15} {recovery_rate:>6.2f}%")
        else:
            print(f"{method:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GSM-8K Relearn Attack Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model with specific unlearn method
  python scripts/analyze_gsm8k_results.py --model Llama-3.2-1B-Instruct --method GradientAscent
  
  # Compare multiple unlearn methods
  python scripts/analyze_gsm8k_results.py --model Llama-3.2-1B-Instruct --compare GradientAscent NPO
  
  # Analyze all available models and methods
  python scripts/analyze_gsm8k_results.py --all
        """
    )
    
    parser.add_argument("--model", type=str, help="Model name (e.g., Llama-3.2-1B-Instruct)")
    parser.add_argument("--method", type=str, help="Unlearn method (e.g., GradientAscent, NPO)")
    parser.add_argument("--compare", nargs="+", help="Compare multiple unlearn methods")
    parser.add_argument("--base-dir", type=str, default="saves/eval", help="Base directory for results")
    parser.add_argument("--all", action="store_true", help="Analyze all available results")
    
    args = parser.parse_args()
    
    if args.all:
        # Find all available results
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"Error: Directory not found: {base_dir}")
            sys.exit(1)
        
        # Scan for available models and methods
        available = {}
        for path in base_dir.glob("gsm8k_*_stage2_unlearn"):
            parts = path.name.replace("gsm8k_", "").replace("_stage2_unlearn", "").rsplit("_", 1)
            if len(parts) == 2:
                model_part, method = parts[0], parts[1]
                if model_part not in available:
                    available[model_part] = []
                available[model_part].append(method)
        
        if not available:
            print("No results found. Have you run the experiments?")
            sys.exit(0)
        
        for model, methods in available.items():
            if len(methods) > 1:
                compare_methods(model, methods, args.base_dir)
            else:
                analyze_model_results(model, methods[0], args.base_dir)
    
    elif args.compare:
        if not args.model:
            print("Error: --model is required when using --compare")
            sys.exit(1)
        compare_methods(args.model, args.compare, args.base_dir)
    
    elif args.model and args.method:
        analyze_model_results(args.model, args.method, args.base_dir)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
