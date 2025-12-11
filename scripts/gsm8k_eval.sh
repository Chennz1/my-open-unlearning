#!/bin/bash

# GSM-8K Evaluation Script for Relearn Attack Study
# Evaluates fine-tuned models on GSM-8K dataset

models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "phi-1_5"
)

########################################################################################################################
########################################### Evaluate GSM-8K Fine-tuned Models ##########################################
########################################################################################################################

for model in "${models[@]}"; do
    echo "=========================================="
    echo "Evaluating ${model} on GSM-8K test set"
    echo "=========================================="
    
    # Evaluate the fine-tuned model
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${model}_relearn \
        eval=gsm8k \
        eval.gsm8k.subset="test" \
        eval.gsm8k.num_samples=null \
        paths.output_dir=saves/eval/gsm8k_${model}_relearn \
        seed=42
    
    echo "Evaluation completed for ${model}"
    echo "Results saved at: saves/eval/gsm8k_${model}_relearn"
    echo ""
done

echo "=========================================="
echo "All GSM-8K evaluations completed!"
echo "=========================================="
