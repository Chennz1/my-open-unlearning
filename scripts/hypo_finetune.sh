#!/bin/bash

# Task Vector Consistency Experiment - Multiple Fine-tuning Runs
# This script fine-tunes models multiple times with different seeds to verify task vector consistency
# Hypothesis: Task vectors from the same base model and training data should have consistent directions

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Model configurations
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "phi-1_5"
    # "Llama-2-7b-chat-hf"
)

# Training hyperparameters
per_device_train_batch_size=4
gradient_accumulation_steps=4  # Effective batch size = 4 * 4 = 16
learning_rate=1e-5
num_epochs=5

# Experiment configuration for task vector consistency
num_runs=5              # Number of independent runs with different seeds
seed_start=42           # Starting seed value
seed_step=100           # Increment for each run (42, 142, 242, 342, 442)

########################################################################################################################
########################### Task Vector Consistency Experiment - Multiple Independent Runs ######################
########################################################################################################################

# model_path="saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO"
model_path="/cnz/data/ms-home/Llama-3___2-1B-Instruct"
method="hypo_consistency"

echo "========================================================================"
echo "Task Vector Consistency Experiment"
echo "========================================================================"
echo "Configuration:"
echo "  - Base model: ${model_path}"
echo "  - Number of independent runs: ${num_runs}"
echo "  - Seed range: ${seed_start} to $((seed_start + (num_runs - 1) * seed_step))"
echo "  - Learning rate: ${learning_rate}"
echo "  - Epochs: ${num_epochs}"
echo "========================================================================"
echo ""

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Model: ${model}"
    echo "=========================================="
    
    # Run multiple independent training runs with different seeds
    for ((run=0; run<num_runs; run++)); do
        # Calculate seed for this run
        current_seed=$((seed_start + run * seed_step))
        
        echo ""
        echo "----------------------------------------"
        echo "Run ${run}/${num_runs} - Seed: ${current_seed}"
        echo "----------------------------------------"
        
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py experiment=finetune/gsm8k/default.yaml \
            task_name=gsm8k_${model}_${method}_run${run}_seed${current_seed} \
            model=${model} \
            data/datasets@data.train=GSM8K_train \
            data/datasets@data.eval=GSM8K_test \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.learning_rate=${learning_rate} \
            trainer.args.num_train_epochs=${num_epochs} \
            trainer.args.seed=${current_seed} \
            +trainer.args.data_seed=${current_seed} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.save_strategy="epoch" \
            trainer.args.output_dir="saves/finetune/consistency_exp/${model}/run${run}_seed${current_seed}"
        
        echo "Run ${run} completed - Model saved at: saves/finetune/consistency_exp/${model}/run${run}_seed${current_seed}"
    done
    
    echo ""
    echo "=========================================="
    echo "All ${num_runs} runs completed for ${model}"
    echo "=========================================="
    echo ""
done

echo "========================================================================"
echo "Task Vector Consistency Experiment Completed!"
echo "========================================================================"
echo "Next steps:"
echo "  1. Compute task vectors: τ_i = θ_fti - θ_base for each run"
echo "  2. Analyze consistency: Calculate cosine similarity between all pairs"
echo "  3. Run PCA analysis to find principal direction"
echo "  4. Visualize results"
echo ""
echo "Model locations:"
for ((run=0; run<num_runs; run++)); do
    current_seed=$((seed_start + run * seed_step))
    echo "  Run ${run} (seed ${current_seed}): saves/finetune/consistency_exp/\${model}/run${run}_seed${current_seed}"
done
echo "========================================================================"
