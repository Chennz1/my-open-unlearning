#!/bin/bash

# Hyperparameter search script for model merging and evaluation
# Searches over alpha (1.0-1.2) and beta (0.7-1.0)

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Llama-2-7b-chat-hf"
trainer="SimNPO"
experiment="unlearn/tofu/default.yaml"
forget_split="forget10"
holdout_split="holdout10"
retain_split="retain90"

# Model paths
tokenizer_path="/cnz/data/ms-home/Llama-2-7b-chat-hf"
orig_model_path="/cnz/data/ms-home/Llama-2-7b-chat-hf"
ft_model_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full"
forget_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_forget10_epoch5"
retain_mimic_base="/cnz/data/project/my-open-unlearning/saves/finetune"

# Output base directory
output_base="/cnz/data/project/my-open-unlearning/saves/unlearn/ratio_search_forget10_7b_mlp_e5"
mkdir -p ${output_base}

# CUDA device
device=0

# Fixed alpha and beta
alpha=1.0
beta=1.0

# Ratio range to explore
ratios=(100)

# Log file
log_file="${output_base}/search_log.txt"
echo "Ratio Search Started at $(date)" > ${log_file}
echo "Fixed alpha: ${alpha}" >> ${log_file}
echo "Fixed beta: ${beta}" >> ${log_file}
echo "Ratio range: ${ratios[@]}" >> ${log_file}
echo "========================================" >> ${log_file}

# Counter for progress
total_combinations=${#ratios[@]}
current=0

# Main search loop
for ratio in "${ratios[@]}"; do
    current=$((current + 1))
    echo ""
    echo "========================================"
    echo "Progress: ${current}/${total_combinations}"
    echo "Testing ratio=${ratio} (alpha=${alpha}, beta=${beta})"
    echo "========================================"
    
    # Format ratio for path (replace . with _)
    ratio_str=$(echo ${ratio})
    
    # Construct retain_mimic_path for current ratio
    retain_mimic_path="${retain_mimic_base}/tofu_Llama-2-7b-chat-hf_retain90_mimic_${ratio_str}"
    
    # Check if retain_mimic_path exists
    if [ ! -d "${retain_mimic_path}" ]; then
        echo "Warning: retain_mimic_path does not exist: ${retain_mimic_path}" | tee -a ${log_file}
        echo "Skipping ratio=${ratio}"
        continue
    fi
    
    # Create temporary model path with ratio in the name
    temp_model_path="${output_base}/model_ratio${ratio_str}"
    
    # Log current experiment
    echo "[$(date)] Starting ratio=${ratio}" >> ${log_file}
    echo "  retain_mimic_path: ${retain_mimic_path}" >> ${log_file}
    
    # Step 1: Merge models with current ratio's retain_mimic_path
    echo "Step 1/3: Merging models..."
    CUDA_VISIBLE_DEVICES=${device} python scripts/effiecient_model_merge_muse_7b.py \
        --tokenizer_path ${tokenizer_path} \
        --orig_model_path ${orig_model_path} \
        --ft_model_path ${ft_model_path} \
        --forget_path ${forget_path} \
        --retain_mimic_path ${retain_mimic_path} \
        --save_path ${temp_model_path} \
        --device ${device} \
        --alpha ${alpha} \
        --beta ${beta}
    
    if [ $? -ne 0 ]; then
        echo "Error: Model merge failed for ratio=${ratio}" | tee -a ${log_file}
        continue
    fi
    
    echo "Model merged successfully"
    
    # Step 2: Evaluate the merged model
    echo "Step 2/3: Evaluating model..."
    task_name=tofu_${model}_${forget_split}_${trainer}_ratio${ratio_str}
    
    CUDA_VISIBLE_DEVICES=${device} python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${temp_model_path} \
        paths.output_dir=${temp_model_path}/evals \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for ratio=${ratio}" | tee -a ${log_file}
    else
        echo "Evaluation completed successfully"
        echo "[$(date)] Completed ratio=${ratio}" >> ${log_file}
    fi
    
    # Step 3: Delete model weights, keep only evals
    echo "Step 3/3: Cleaning up model weights..."
    if [ -d "${temp_model_path}" ]; then
        # Keep the evals directory
        if [ -d "${temp_model_path}/evals" ]; then
            # Move evals to a safe location
            mv ${temp_model_path}/evals ${output_base}/evals_ratio${ratio_str}
        fi
        
        # Remove the entire model directory
        rm -rf ${temp_model_path}
        echo "Model weights deleted, evals preserved"
    fi
    
    echo "Finished ratio=${ratio}"
    echo ""
done

echo ""
echo "========================================"
echo "Ratio search completed!"
echo "Total ratios tested: ${total_combinations}"
echo "Results saved in: ${output_base}"
echo "========================================"
echo "" >> ${log_file}
echo "Ratio Search Completed at $(date)" >> ${log_file}
echo "Total ratios tested: ${total_combinations}" >> ${log_file}

# Create a summary of all results
echo "Creating results summary..."
summary_file="${output_base}/results_summary.txt"
echo "Ratio Search Results Summary" > ${summary_file}
echo "======================================" >> ${summary_file}
echo "Model: ${model}" >> ${summary_file}
echo "Forget split: ${forget_split}" >> ${summary_file}
echo "Fixed alpha: ${alpha}" >> ${summary_file}
echo "Fixed beta: ${beta}" >> ${summary_file}
echo "Ratio range: ${ratios[@]}" >> ${summary_file}
echo "======================================" >> ${summary_file}
echo "" >> ${summary_file}

# List all evaluation results
for ratio in "${ratios[@]}"; do
    ratio_str=$(echo ${ratio} | tr '.' '_')
    evals_dir="${output_base}/evals_ratio${ratio_str}"
    if [ -d "${evals_dir}" ]; then
        echo "Ratio=${ratio}:" >> ${summary_file}
        # Try to extract key metrics if TOFU_EVAL.json exists
        if [ -f "${evals_dir}/TOFU_EVAL.json" ]; then
            echo "  Evaluation file found: ${evals_dir}/TOFU_EVAL.json" >> ${summary_file}
        else
            echo "  Warning: Evaluation file not found" >> ${summary_file}
        fi
        echo "" >> ${summary_file}
    fi
done

echo "Summary saved to: ${summary_file}"
echo "All done!"
