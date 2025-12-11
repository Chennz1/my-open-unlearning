#!/bin/bash

# Iterative Task Vector Experiment Script
# This script runs the iterative task vector approach for different forget splits

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
BASE_MODEL="saves/finetune/tofu_Llama-3.2-1B-Instruct_full"
OUTPUT_BASE_DIR="saves/iterative_task_vector"
DEVICE="0"
NUM_ITERATIONS=3
OVERFORGET_EPOCHS=8
ALPHA=1.0

# Experiment splits to test
splits=(
    "forget01 holdout01 retain99"
    "forget10 holdout10 retain90"
)

echo "========================================"
echo "ITERATIVE TASK VECTOR EXPERIMENTS"
echo "========================================"
echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Number of iterations: $NUM_ITERATIONS"
echo "Overforget epochs: $OVERFORGET_EPOCHS"
echo "Device: $DEVICE"
echo "Alpha: $ALPHA"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_BASE_DIR

# Run experiments for each split
for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    echo ""
    echo "========================================"
    echo "RUNNING EXPERIMENT: $forget_split"
    echo "========================================"
    echo "Forget split: $forget_split"
    echo "Holdout split: $holdout_split"
    echo "Retain split: $retain_split"
    echo "========================================"
    
    # Create experiment-specific output directory
    exp_output_dir="${OUTPUT_BASE_DIR}/${forget_split}"
    mkdir -p $exp_output_dir
    
    # Run the iterative task vector experiment
    echo "Starting iterative task vector experiment..."
    python iterative_task_vector.py \
        --base_model $BASE_MODEL \
        --forget_split $forget_split \
        --holdout_split $holdout_split \
        --retain_split $retain_split \
        --num_iterations $NUM_ITERATIONS \
        --overforget_epochs $OVERFORGET_EPOCHS \
        --output_dir $exp_output_dir \
        --device $DEVICE \
        --alpha $ALPHA
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment completed successfully for $forget_split"
    else
        echo "✗ Experiment failed for $forget_split"
    fi
    
    echo "Results saved to: $exp_output_dir"
done

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================"
echo "Results directory: $OUTPUT_BASE_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
python analyze_iterative_results.py --results_dir $OUTPUT_BASE_DIR

echo "Summary report generated."
echo "Check the results in: $OUTPUT_BASE_DIR"