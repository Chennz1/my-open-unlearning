#!/bin/bash

# Quick Test Script for Iterative Task Vector
# This script runs a single iteration to test the implementation

echo "========================================"
echo "ITERATIVE TASK VECTOR - QUICK TEST"
echo "========================================"

# Configuration for testing
BASE_MODEL="saves/finetune/tofu_Llama-3.2-1B-Instruct_full"
OUTPUT_DIR="saves/iterative_task_vector_test"
DEVICE="0"
FORGET_SPLIT="forget01"
HOLDOUT_SPLIT="holdout01"
RETAIN_SPLIT="retain99"

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model not found at $BASE_MODEL"
    echo "Available models in saves/finetune/:"
    ls -la saves/finetune/ | grep tofu
    exit 1
fi

echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Forget split: $FORGET_SPLIT"
echo "Device: $DEVICE"
echo "========================================"

# Run a quick test with just 1 iteration and fewer epochs
echo "Running quick test with 1 iteration..."

python iterative_task_vector.py \
    --base_model $BASE_MODEL \
    --forget_split $FORGET_SPLIT \
    --holdout_split $HOLDOUT_SPLIT \
    --retain_split $RETAIN_SPLIT \
    --num_iterations 1 \
    --overforget_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --alpha 1.0

if [ $? -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Show results summary
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "Results preview:"
        python -c "
import json
with open('$OUTPUT_DIR/results.json', 'r') as f:
    data = json.load(f)
if data['iterations']:
    iter_data = data['iterations'][0]
    print(f'Iteration 1 completed in {iter_data[\"time_seconds\"]:.1f} seconds')
    if iter_data['evaluations']['updated_combined'] and 'forget_quality' in iter_data['evaluations']['updated_combined']:
        fq = iter_data['evaluations']['updated_combined']['forget_quality']
        print(f'Final model forget quality: {fq:.4f}')
    else:
        print('Evaluation data not available')
else:
    print('No iteration data found')
"
    fi
    
    echo ""
    echo "If the test was successful, you can run the full experiment with:"
    echo "bash scripts/run_iterative_task_vector.sh"
    
else
    echo "✗ Quick test failed!"
    echo "Check the error messages above."
    exit 1
fi