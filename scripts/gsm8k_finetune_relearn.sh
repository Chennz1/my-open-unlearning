#!/bin/bash

# GSM-8K Fine-tuning Script for Relearn Attack Study
# This script fine-tunes models on GSM-8K dataset to study relearn attack effectiveness

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Model configurations
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "phi-1_5"
)

# Training hyperparameters
per_device_train_batch_size=4
gradient_accumulation_steps=4  # Effective batch size = 4 * 4 = 16
learning_rate=1e-5
num_epochs=5

########################################################################################################################
########################################### GSM-8K Fine-tuning for Relearn Attack Study ################################
########################################################################################################################

# model_path="saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO"
model_path="vectors/test_model_10"


for model in "${models[@]}"; do
    echo "=========================================="
    echo "Fine-tuning ${model} on GSM-8K dataset"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/gsm8k/default.yaml \
        task_name=gsm8k_${model}_relearn \
        model=${model} \
        data/datasets@data.train=GSM8K_train \
        data/datasets@data.eval=GSM8K_test \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.learning_rate=${learning_rate} \
        trainer.args.num_train_epochs=${num_epochs} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy="epoch"
    
    echo "Fine-tuning completed for ${model}"
    echo "Model saved at: saves/finetune/gsm8k_${model}_relearn"
    echo ""
done

echo "=========================================="
echo "All GSM-8K fine-tuning jobs completed!"
echo "=========================================="
