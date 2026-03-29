#!/bin/bash

# Hyperparameter search script for model merging and evaluation
# Searches over alpha (1.0-1.2) and beta (0.7-1.0)

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="phi-1_5"
trainer="SimNPO"
experiment="unlearn/tofu/default.yaml"
forget_split="forget05"
holdout_split="holdout05"
retain_split="retain95"

# Model paths
tokenizer_path="/cnz/data/hf-home/hub/hub/models--microsoft--phi-1_5/snapshots/77aa61eeac94fbf33d492b9f2744c98b42d5b5eb"
orig_model_path="/cnz/data/hf-home/hub/hub/models--microsoft--phi-1_5/snapshots/77aa61eeac94fbf33d492b9f2744c98b42d5b5eb"
ft_model_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_phi-1_5_full"
forget_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_phi-1_5_forget05_epoch5"
retain_mimic_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_phi-1_5_retain95_mimic_epoch5"

# Output base directory
output_base="/cnz/data/project/my-open-unlearning/saves/unlearn/hyperparam_search_forget05_phi_mlp_e5"
mkdir -p ${output_base}

# CUDA device
device=0

# Hyperparameter ranges
# Alpha:
alphas=(0.8 1.0 1.2)
# Beta:
betas=(0.8 1.0 1.2)

# Log file
log_file="${output_base}/search_log.txt"
echo "Hyperparameter Search Started at $(date)" > ${log_file}
echo "Alpha range: ${alphas[@]}" >> ${log_file}
echo "Beta range: ${betas[@]}" >> ${log_file}
echo "========================================" >> ${log_file}

# Counter for progress
total_combinations=$((${#alphas[@]} * ${#betas[@]}))
current=0

# Main search loop
for alpha in "${alphas[@]}"; do
    for beta in "${betas[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================"
        echo "Progress: ${current}/${total_combinations}"
        echo "Testing alpha=${alpha}, beta=${beta}"
        echo "========================================"
        
        # Create temporary model path with alpha and beta in the name
        temp_model_path="${output_base}/model_a${alpha}_b${beta}"
        
        # Log current experiment
        echo "[$(date)] Starting alpha=${alpha}, beta=${beta}" >> ${log_file}
        
        # Step 1: Merge models with current alpha and beta
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
            echo "Error: Model merge failed for alpha=${alpha}, beta=${beta}" | tee -a ${log_file}
            continue
        fi
        
        echo "Model merged successfully"
        
        # Step 2: Evaluate the merged model
        echo "Step 2/3: Evaluating model..."
        task_name=tofu_${model}_${forget_split}_${trainer}_a${alpha}_b${beta}
        
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
            echo "Error: Evaluation failed for alpha=${alpha}, beta=${beta}" | tee -a ${log_file}
        else
            echo "Evaluation completed successfully"
            echo "[$(date)] Completed alpha=${alpha}, beta=${beta}" >> ${log_file}
        fi
        
        # # Step 3: Delete model weights, keep only evals
        # echo "Step 3/3: Cleaning up model weights..."
        # if [ -d "${temp_model_path}" ]; then
        #     # Keep the evals directory
        #     if [ -d "${temp_model_path}/evals" ]; then
        #         # Move evals to a safe location
        #         mv ${temp_model_path}/evals ${output_base}/evals_a${alpha}_b${beta}
        #     fi
            
        #     # Remove the entire model directory
        #     rm -rf ${temp_model_path}
        #     echo "Model weights deleted, evals preserved"
        # fi
        
        echo "Finished alpha=${alpha}, beta=${beta}"
        echo ""
    done
done

echo ""
echo "========================================"
echo "Hyperparameter search completed!"
echo "Total combinations tested: ${total_combinations}"
echo "Results saved in: ${output_base}"
echo "========================================"
echo "" >> ${log_file}
echo "Hyperparameter Search Completed at $(date)" >> ${log_file}
echo "Total combinations tested: ${total_combinations}" >> ${log_file}

# Create a summary of all results
echo "Creating results summary..."
summary_file="${output_base}/results_summary.txt"
echo "Hyperparameter Search Results Summary" > ${summary_file}
echo "======================================" >> ${summary_file}
echo "Model: ${model}" >> ${summary_file}
echo "Forget split: ${forget_split}" >> ${summary_file}
echo "Alpha range: ${alphas[@]}" >> ${summary_file}
echo "Beta range: ${betas[@]}" >> ${summary_file}
echo "======================================" >> ${summary_file}
echo "" >> ${summary_file}

# List all evaluation results
for alpha in "${alphas[@]}"; do
    for beta in "${betas[@]}"; do
        evals_dir="${output_base}/evals_a${alpha}_b${beta}"
        if [ -d "${evals_dir}" ]; then
            echo "Alpha=${alpha}, Beta=${beta}:" >> ${summary_file}
            # Try to extract key metrics if TOFU_EVAL.json exists
            if [ -f "${evals_dir}/TOFU_EVAL.json" ]; then
                echo "  Evaluation file found: ${evals_dir}/TOFU_EVAL.json" >> ${summary_file}
            else
                echo "  Warning: Evaluation file not found" >> ${summary_file}
            fi
            echo "" >> ${summary_file}
        fi
    done
done

echo "Summary saved to: ${summary_file}"
echo "All done!"
