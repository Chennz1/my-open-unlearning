#!/bin/bash

# Quick Start Example for GSM-8K Relearn Attack Study
# This is a minimal example to test the setup

echo "GSM-8K Relearn Attack Study - Quick Start Example"
echo "=================================================="
echo ""

# Configuration
MODEL="Llama-3.2-1B-Instruct"
METHOD="GradientAscent"

echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Unlearn Method: ${METHOD}"
echo ""

# Check if model exists
echo "Step 1: Checking dependencies..."
python -c "from transformers import AutoTokenizer; print('✓ Transformers installed')" || exit 1
python -c "from datasets import load_dataset; print('✓ Datasets installed')" || exit 1
echo ""

# Stage 1: Fine-tune on GSM-8K (simplified)
echo "Step 2: Fine-tuning on GSM-8K (1 epoch for testing)..."
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    experiment=finetune/gsm8k/default.yaml \
    task_name=gsm8k_${MODEL}_example \
    model=${MODEL} \
    trainer.args.num_train_epochs=1 \
    trainer.args.max_steps=100 \
    trainer.args.save_strategy="no" \
    trainer.args.evaluation_strategy="no" \
    paths.output_dir=saves/finetune/gsm8k_${MODEL}_example

if [ $? -eq 0 ]; then
    echo "✓ Fine-tuning completed"
else
    echo "✗ Fine-tuning failed"
    exit 1
fi
echo ""

# Stage 2: Evaluate
echo "Step 3: Evaluating on GSM-8K test set (100 samples for testing)..."
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${MODEL}_example \
    eval=gsm8k \
    eval.gsm8k.subset="test" \
    eval.gsm8k.num_samples=100 \
    paths.output_dir=saves/eval/gsm8k_${MODEL}_example \
    seed=42

if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed"
    echo ""
    echo "Results:"
    cat saves/eval/gsm8k_${MODEL}_example/GSM8K_SUMMARY.json
else
    echo "✗ Evaluation failed"
    exit 1
fi
echo ""

echo "=================================================="
echo "Quick start example completed!"
echo ""
echo "Next steps:"
echo "1. Review the results in: saves/eval/gsm8k_${MODEL}_example/"
echo "2. Run full pipeline: bash scripts/gsm8k_relearn_attack_pipeline.sh"
echo "3. Analyze results: python scripts/analyze_gsm8k_results.py --all"
echo "=================================================="
