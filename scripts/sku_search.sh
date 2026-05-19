#!/bin/bash

set -u

# Alpha sweep for the final SKU task-vector negation step inside open-unlearning.
# This script only performs model fusion and keeps the interface close to the
# existing search_tv / efficient_task_vector workflow.

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

model="Llama-2-7b-chat-hf"
forget_split="forget10"

tokenizer_path="/cnz/data/ms-home/Llama-2-7b-chat-hf"
orig_model_path="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full"
acquired_model_path="/cnz/data/project/my-open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_${forget_split}_SKU_epoch2"

output_base="/cnz/data/project/my-open-unlearning/saves/unlearn/tv_sku_${forget_split}_${model}"
mkdir -p "${output_base}"

device=0
filter_type="all"
alphas=(1.0)

log_file="${output_base}/search_log.txt"
echo "SKU task-vector search started at $(date)" > "${log_file}"
echo "Alpha range: ${alphas[*]}" >> "${log_file}"
echo "Filter type: ${filter_type}" >> "${log_file}"
echo "========================================" >> "${log_file}"

total_combinations=$((${#alphas[@]}))
current=0

for alpha in "${alphas[@]}"; do
    current=$((current + 1))
    echo ""
    echo "========================================"
    echo "Progress: ${current}/${total_combinations}"
    echo "Testing alpha=${alpha}"
    echo "========================================"

    temp_model_path="${output_base}"
    echo "[$(date)] Starting alpha=${alpha}" >> "${log_file}"

    CUDA_VISIBLE_DEVICES=${device} python scripts/sku_merge.py \
        --tokenizer_path "${tokenizer_path}" \
        --orig_model_path "${orig_model_path}" \
        --acquired_model_path "${acquired_model_path}" \
        --save_path "${temp_model_path}" \
        --alpha "${alpha}" \
        --filter_type "${filter_type}" \
        --device "${device}"

    if [ $? -ne 0 ]; then
        echo "Error: Model merge failed for alpha=${alpha}" | tee -a "${log_file}"
        continue
    fi

    echo "Model merged successfully"
    echo "[$(date)] Completed alpha=${alpha}" >> "${log_file}"
done

echo ""
echo "========================================"
echo "SKU task-vector search completed!"
echo "Total combinations tested: ${total_combinations}"
echo "Results saved in: ${output_base}"
echo "========================================"
echo "" >> "${log_file}"
echo "Search completed at $(date)" >> "${log_file}"
