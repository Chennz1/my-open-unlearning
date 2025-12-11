#!/bin/bash

# Complete Pipeline for GSM-8K Relearn Attack Study
# This script:
# 1. Fine-tunes models on GSM-8K
# 2. Applies unlearning on specific subsets
# 3. Re-learns (relearn attack) using GSM-8K again
# 4. Evaluates at each stage to study the impact

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
)

# Unlearn methods to test
unlearn_methods=(
    # "GradientAscent"
    "NPO"
    # "GradientDifference"
)

########################################################################################################################
########################################### Stage 1: Fine-tune on GSM-8K ##############################################
########################################################################################################################

echo "=========================================="
echo "Stage 1: Fine-tuning models on GSM-8K"
echo "=========================================="

for model in "${models[@]}"; do
    echo "Fine-tuning ${model} on GSM-8K..."
    
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/gsm8k/default.yaml \
        task_name=gsm8k_${model}_stage1 \
        model=${model} \
        trainer.args.num_train_epochs=3 \
        trainer.args.save_strategy="epoch"
    
    # # Evaluate stage 1 model
    # echo "Evaluating stage 1 model..."
    # CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    #     model=${model} \
    #     model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${model}_stage1 \
    #     eval=gsm8k \
    #     paths.output_dir=saves/eval/gsm8k_${model}_stage1 \
    #     seed=42
done

# ########################################################################################################################
# ########################################### Stage 2: Unlearn Specific Knowledge ########################################
# ########################################################################################################################

# echo "=========================================="
# echo "Stage 2: Unlearning specific knowledge"
# echo "=========================================="

# # For this study, we can use TOFU forget set as a proxy for "knowledge to unlearn"
# # You can replace this with your specific unlearn dataset

# for model in "${models[@]}"; do
#     for method in "${unlearn_methods[@]}"; do
#         echo "Applying ${method} unlearning to ${model}..."
        
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#             --config_file configs/accelerate/default_config.yaml \
#             --main_process_port $MASTER_PORT \
#             src/train.py \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${model}_stage1 \
#             trainer=${method} \
#             trainer.args.num_train_epochs=2 \
#             data/datasets@data.forget=TOFU_QA_forget \
#             data/datasets@data.retain=TOFU_QA_retain \
#             data.forget.TOFU_QA_forget.args.hf_args.name=forget10 \
#             data.retain.TOFU_QA_retain.args.hf_args.name=retain90 \
#             task_name=gsm8k_${model}_${method}_stage2_unlearn
        
#         # Evaluate after unlearning (should show degradation on GSM-8K)
#         echo "Evaluating after unlearning..."
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/gsm8k_${model}_${method}_stage2_unlearn \
#             eval=gsm8k \
#             paths.output_dir=saves/eval/gsm8k_${model}_${method}_stage2_unlearn \
#             seed=42
#     done
# done

# ########################################################################################################################
# ########################################### Stage 3: Relearn Attack (Re-fine-tune on GSM-8K) ###########################
# ########################################################################################################################

# echo "=========================================="
# echo "Stage 3: Relearn Attack - Re-fine-tuning on GSM-8K"
# echo "=========================================="

# for model in "${models[@]}"; do
#     for method in "${unlearn_methods[@]}"; do
#         echo "Performing relearn attack on ${model} (unlearned with ${method})..."
        
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#             --config_file configs/accelerate/default_config.yaml \
#             --main_process_port $MASTER_PORT \
#             src/train.py experiment=finetune/gsm8k/default.yaml \
#             task_name=gsm8k_${model}_${method}_stage3_relearn \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/gsm8k_${model}_${method}_stage2_unlearn \
#             trainer.args.num_train_epochs=2 \
#             trainer.args.learning_rate=1e-5 \
#             trainer.args.save_strategy="epoch"
        
#         # Evaluate after relearn attack
#         echo "Evaluating after relearn attack..."
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${model}_${method}_stage3_relearn \
#             eval=gsm8k \
#             paths.output_dir=saves/eval/gsm8k_${model}_${method}_stage3_relearn \
#             seed=42
        
#         # Also evaluate on the forget set to see if forgotten knowledge returns
#         CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/finetune/gsm8k_${model}_${method}_stage3_relearn \
#             forget_split=forget10 \
#             holdout_split=holdout10 \
#             task_name=gsm8k_${model}_${method}_stage3_relearn \
#             paths.output_dir=saves/eval/gsm8k_${model}_${method}_stage3_relearn_tofu
#     done
# done

# ########################################################################################################################
# ########################################### Summary and Analysis #######################################################
# ########################################################################################################################

# echo "=========================================="
# echo "Pipeline completed! Summary of stages:"
# echo "=========================================="
# echo "Stage 1: Fine-tuned models on GSM-8K"
# echo "  - Models: ${models[@]}"
# echo "  - Location: saves/finetune/gsm8k_*_stage1"
# echo ""
# echo "Stage 2: Applied unlearning methods"
# echo "  - Methods: ${unlearn_methods[@]}"
# echo "  - Location: saves/unlearn/gsm8k_*_stage2_unlearn"
# echo ""
# echo "Stage 3: Relearn attack by re-fine-tuning on GSM-8K"
# echo "  - Location: saves/finetune/gsm8k_*_stage3_relearn"
# echo ""
# echo "All evaluations saved in: saves/eval/"
# echo ""
# echo "To analyze results, compare GSM-8K accuracy across stages:"
# echo "  - saves/eval/gsm8k_*_stage1/GSM8K_SUMMARY.json"
# echo "  - saves/eval/gsm8k_*_stage2_unlearn/GSM8K_SUMMARY.json"
# echo "  - saves/eval/gsm8k_*_stage3_relearn/GSM8K_SUMMARY.json"
# echo "=========================================="
