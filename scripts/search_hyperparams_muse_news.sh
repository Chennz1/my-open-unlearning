#!/bin/bash

# Hyperparameter search script for MUSE model merging and evaluation
# Searches over alpha and beta

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Llama-2-7b-hf"
trainer="SimNPO" # Or whatever trainer name is appropriate for the task vector method
experiment="eval/muse/default.yaml"
data_split="News"

# Model paths
tokenizer_path="/cnz/data/ms-home/Llama-2-7b-hf"
orig_model_path="/cnz/data/ms-home/Llama-2-7b-hf"
ft_model_path="/cnz/data/hf-home/hub/models--muse-bench--MUSE-news_target/snapshots/a2f39769e9a0b98ec1cdd12f65e9962502208935"
forget_path="/cnz/data/project/my-open-unlearning/saves/finetune/muse_Llama-2-7b-hf_News_forget_e10"
retain_mimic_path="/cnz/data/project/my-open-unlearning/saves/finetune/muse_Llama-2-7b-hf_News_retain1_e10"

# Output base directory
output_base="/cnz/data/project/my-open-unlearning/saves/unlearn/hyperparam_search_muse_news_7b_mlp_e10"
mkdir -p ${output_base}

# CUDA device
device=0

# Hyperparameter ranges
# Alpha:
alphas=(1.6 1.8 2.0)
# Beta:
betas=(0.2 0.8 1.4)

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
        task_name=muse_${model}_${data_split}_${trainer}_a${alpha}_b${beta}
        
        CUDA_VISIBLE_DEVICES=${device} python src/eval.py \
            experiment=${experiment} \
            data_split=${data_split} \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${temp_model_path} \
            paths.output_dir=${temp_model_path}/evals \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        
        if [ $? -ne 0 ]; then
            echo "Error: Evaluation failed for alpha=${alpha}, beta=${beta}" | tee -a ${log_file}
        else
            echo "Evaluation completed successfully"
            echo "[$(date)] Completed alpha=${alpha}, beta=${beta}" >> ${log_file}
        fi
        
        # Step 3: Delete model weights, keep only evals
        echo "Step 3/3: Cleaning up model weights..."
        if [ -d "${temp_model_path}" ]; then
            # Keep the evals directory
            if [ -d "${temp_model_path}/evals" ]; then
                # Move evals to a safe location
                mv ${temp_model_path}/evals ${output_base}/evals_a${alpha}_b${beta}
            fi
            
            # Remove the entire model directory
            rm -rf ${temp_model_path}
            echo "Model weights deleted, evals preserved"
        fi
        
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
echo "Data split: ${data_split}" >> ${summary_file}
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
            # Try to extract key metrics if MUSE_EVAL.json exists
            if [ -f "${evals_dir}/MUSE_EVAL.json" ]; then
                echo "  Evaluation file found: ${evals_dir}/MUSE_EVAL.json" >> ${summary_file}
            else
                echo "  Warning: Evaluation file not found" >> ${summary_file}
            fi
            echo "" >> ${summary_file}
        fi
    done
done

echo "Summary saved to: ${summary_file}"
echo "All done!"
